#pragma once

#include "../flatnav/DistanceInterface.h"
#include "CentroidsGenerator.h"
#include "Utils.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

namespace flatnav::quantization {

using flatnav::quantization::CentroidsGenerator;

template <typename n_bits_t> struct PQCodeManager {
  // This is an array that represents a quantization code for
  // a given vector. For instance, if x = [x_0, ..., x_{m-1}]
  // is subdivided into 8 subvectors, each of size m/8, we will
  // get a code for each subvector. This array will hold the codes
  // sequentially. That means, it will be of size m/8 as well.
  //
  // NOTE: code here means the index of the local centroid that
  // minimizes the (squared) distance between a given subvector and itself.
  n_bits_t *code;
  n_bits_t *start;

  // Indicates if the code manager has already been redirected to the
  // start of the encoding so that we don't do this more than once (to
  // avoid segfaults while decoding).
  bool code_manager_already_set_to_start;

  PQCodeManager(uint8_t *code, uint32_t nbits)
      : code(reinterpret_cast<n_bits_t *>(code)),
        start(reinterpret_cast<n_bits_t *>(code)),
        code_manager_already_set_to_start(false) {
    assert(nbits == 8 * sizeof(n_bits_t));
  }

  void encode(uint64_t index) {
    *code = static_cast<n_bits_t>(index);
    code++;
  }

  uint64_t decode() {
    uint64_t decoded = static_cast<uint64_t>(*code);
    code++;
    return decoded;
  }

  void jumpToStart() {
    if (!code_manager_already_set_to_start) {
      code = start;
      code_manager_already_set_to_start = true;
    }
  }
};

/** This is the main class definition for a regular Product Quantizer
 * The implementation is simultaneously inspired by FAISS(https://faiss.ai)
 * and the paper Cache Locality is not Enough: High-Performance Nearest
 * Neighbor Search with Product Quantization Fast Scan
 * http://www.vldb.org/pvldb/vol9/p288-andre.pdf
 *
 */

template <typename dist_t> class ProductQuantizer {
  // Represents the block size used in ProductQuantizer::computePQCodes
  static const uint64_t BLOCK_SIZE = 256 * 1024;

  // This heap array maintains an ordering of pairs of computed distances
  // (float) and their assigned ID's (int64_t).
  // TODO: Implement a modular heap class that can both be used
  // in Flatnav's Index and Quantization
  typedef std::priority_queue<std::pair<float, int64_t>,
                              std::vector<std::pair<float, int64_t>>>
      MaxHeap;

public:
  /** PQ Constructor.
   *
   * @param dim      dimensionality of the input vectors
   * @param M        number of subquantizers
   * @param nbits    number of bit per subvector index
   *
   * TODO: Only pass the distance interface to the underlying index.
   * This will be possible once the PQ integration with the flatnav
   * index is complete.
   */
  ProductQuantizer(std::unique_ptr<DistanceInterface<dist_t>> dist,
                   uint32_t dim, uint32_t M, uint32_t nbits)
      : _num_subquantizers(M), _num_bits(nbits), _distance(std::move(dist)),
        _train_type(TrainType::DEFAULT) {

    if (dim % _num_subquantizers) {
      throw std::invalid_argument("The dataset dimension must be a multiple of "
                                  "the desired number of sub-quantizers.");
    }
    _code_size = (_num_bits * 8 + 7) / 8;
    _subvector_dim = dim / _num_subquantizers;
    _subq_centroids_count = 1 << _num_bits;
    _centroids.resize(_subq_centroids_count * dim);
  }

  // Return a pointer to the centroids associated with a given subvector
  const float *getCentroids(uint32_t subvector_index, uint32_t i) const {
    auto index = (subvector_index * _subq_centroids_count + i) * _subvector_dim;
    return &_centroids[index];
  }

  void setParameters(const float *centroids_, int m) {
    float *centroids = const_cast<float *>(getCentroids(m, 0));
    auto bytes_to_copy =
        _subq_centroids_count * _subvector_dim * sizeof(_centroids[0]);

    std::memcpy(centroids, centroids_, bytes_to_copy);
  }

  /**
   * @brief Quantize a single vector with PQ
   *
   * @param vector
   * @param code
   */
  void computePQCode(const float *vector, uint8_t *code) const {
    std::vector<float> distances(_subq_centroids_count);

    // TODO check whether this const_cast does not cause any issues
    PQCodeManager<uint8_t> code_manager(
        /* code = */ reinterpret_cast<uint8_t *>(const_cast<float *>(vector)),
        /* nbits = */ 8);

    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      const float *subvector = vector + (m * _subvector_dim);
      uint64_t minimizer_index = squaredL2WithKNeighbors(
          /* distances_tmp_buffer = */ distances.data(), /* x = */ subvector,
          /* y = */ getCentroids(m, 0), /* dim = */ _subvector_dim,
          /* target_set_size = */ _subq_centroids_count);

      code_manager.encode(minimizer_index);
    }
  }

  /**
   * @brief Quantize multiple vectors with PQ.
   * We recursively extract smaller chunks and then process vectors in the
   * chunks in parallel.
   *
   * @param vectors        pointer to the vectors to be quantized
   * @param codes          quantization codes
   * @param n              total number of vectors
   */
  void computePQCodes(const float *vectors, uint8_t *codes, uint64_t n) const {
    // process by blocks to avoid using too much RAM

    auto dim = _subvector_dim * _num_subquantizers;
    if (n > BLOCK_SIZE) {
      for (uint64_t i0 = 0; i0 < n; i0 += BLOCK_SIZE) {
        auto i1 = std::min(i0 + BLOCK_SIZE, n);
        computePQCodes(vectors + (dim * i0), codes + (_code_size * i0),
                       i1 - i0);
      }
      return;
    }

#pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
      computePQCode(vectors + (i * dim), codes + (i * _code_size));
    }
  }

  /**
   * @brief Trains a product quantizer on a given set of vectors
   *
   * @param vectors          Vectors to use for quantization
   * @param n                Number of vectors
   */
  void train(const float *vectors, uint64_t n) {

    CentroidsGenerator centroids_generator(
        /* dim = */ _subvector_dim,
        /* num_centroids = */ _subq_centroids_count);

    if (_train_type == TrainType::SHARED) {
      centroids_generator.generateCentroids(
          /* vectors = */ vectors, /* vec_weights = */ NULL,
          /* n = */ n * _num_subquantizers);

      for (uint32_t m = 0; m < _num_subquantizers; m++) {
        setParameters(/* centroids_ = */ centroids_generator.centroids().data(),
                      /* m = */ m);
      }
      return;
    }

    TrainType final_train_type = _train_type;

    if (_train_type == TrainType::HYPERCUBE ||
        _train_type == TrainType::HYPERCUBE_PCA) {
      if (_subvector_dim < _num_bits) {
        final_train_type = TrainType::DEFAULT;
        std::cout << "[pq-train-warning] cannot train hypercube with num "
                     "bits greater than subvector dimension"
                  << std::endl;
      }
    }

    // float *slice = new float[n * _subvector_dim];
    std::shared_ptr<float[]> slice(new float[n * _subvector_dim]);

    auto dim = _subvector_dim * _num_subquantizers;
    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
        std::memcpy(slice.get() + (vec_index * _subvector_dim),
                    vectors + (vec_index * dim) + (m * _subvector_dim),
                    _subvector_dim * sizeof(float));
      }

      // Do this when we have initialization for the centroids
      if (final_train_type != TrainType::DEFAULT) {
        centroids_generator.centroids().resize(_subq_centroids_count *
                                               _subvector_dim);
      }

      switch (final_train_type) {
      case TrainType::HYPERCUBE:
        initHypercube(
            /* dim = */ _subvector_dim, /* num_bits = */ _num_bits, /* n = */ n,
            /* data = */ slice.get(),
            /* centroids = */ centroids_generator.centroids().data());
        break;

      case TrainType::HOT_START:
        std::memcpy(centroids_generator.centroids().data(), getCentroids(m, 0),
                    _subvector_dim * _subq_centroids_count * sizeof(float));
        break;

      default:;
      }

      // generate the actual centroids
      centroids_generator.generateCentroids(
          /* vectors = */ slice.get(), /* vec_weights = */ NULL, /* n = */ n);

      setParameters(/* centroids_ = */ centroids_generator.centroids().data(),
                    /* m = */ m);
    }
  }

  /**
   * @brief Decode a single vector from a given code. For now we are using
   * `uint8_t` as the default type representing the number of bits per code
   * index.
   *
   * @param code      Code corresponding to the given vector
   * @param vector    Vector to decode
   */
  void decode(const uint8_t *code, float *vector) const {
    // TODO check whether this const_cast does not cause any issues
    PQCodeManager<uint8_t> code_manager(
        /* code = */ reinterpret_cast<uint8_t *>(const_cast<float *>(vector)),
        /* nbits = */ 8);

    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      uint64_t code_ = code_manager.decode();
      std::memcpy(vector + (m * _subvector_dim), getCentroids(m, 0),
                  sizeof(float) * _subvector_dim);
    }
  }

  /**
   * @brief Decode multiple vectors given their respective codes.
   */
  void decode(const uint8_t *code, float *vectors, uint64_t n) const {
    auto dim = _subvector_dim * _num_subquantizers;
    for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
      decode(code + (vec_index * _code_size), vectors + (vec_index * dim));
    }
  }

  /**
   * @brief Compute distance table for a single vector.
   * For more information on distance tables, see
   * http://www.vldb.org/pvldb/vol9/p288-andre.pdf
   *
   * @param x         input vector size d
   * @param dis_table output table, size (_num_subquantizers x
   * _subq_centroids_count)
   */
  void computeDistanceTable(const float *vector, float *dist_table) const {
    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      flatnav::registerSquaredL2Distances(
          /* distances_buffer = */ dist_table + (m * _subq_centroids_count),
          /* x = */ vector + (m * _subvector_dim), /* y = */ getCentroids(m, 0),
          /* dim = */ _subvector_dim,
          /* target_set_size = */ _subq_centroids_count);
    }
  }

  void computeDistanceTables(const float *vectors, float *dist_tables,
                             uint64_t n) const {
    // TODO: Use SIMD
    auto dim = _subvector_dim * _num_subquantizers;
#pragma omp parallel for
    for (uint64_t i = 0; i < n; i++) {
      computeDistanceTable(
          vectors + (i * dim),
          dist_tables + (i * _subq_centroids_count * _num_subquantizers));
    }
  }

  template <typename T>
  void pqEstimatorsFromTables(const ProductQuantizer &pq, const uint8_t *codes,
                              const uint64_t ncodes, const float *dist_table);

  /**
   * @NOTE: This function, in principle, supports searching with multiple
   * queries in parallel. However, we are currently limited by our heap
   * implementation, which does not allow us to process multiple queries at
   * once. So, we will assume for now that `num_queries` = 1.
   *
   */
  void searchUsingDistanceTables(const ProductQuantizer &pq, uint32_t num_bits,
                                 const float *dist_tables, uint64_t num_queries,
                                 const uint8_t *codes, const uint64_t ncodes,
                                 std::shared_ptr<MaxHeap> heap,
                                 bool init_finalize_heap) {}

  /**
   * @brief perform search
   *
   * @param query_vectors        query vectors, size num_queries * d
   * @param num_queries          number of queries
   * @param codes                PQ codes, size ncodes * code_size
   * @param ncodes               nb of vectors
   */
  void search(const float *__restrict query_vectors, uint64_t num_queries,
              const uint8_t *codes, const uint64_t ncodes,
              bool init_finalize_heap) const {

    std::unique_ptr<float[]> dist_tables(
        new float[num_queries * _subq_centroids_count * _num_subquantizers]);

    computeDistanceTables(/* vectors = */ query_vectors,
                          /* dist_tables = */ dist_tables,
                          /* n = */ num_queries);

    searchUsingDistanceTables(
        /* pq = */ *this,
        /* num_bits = */ _num_bits, /* dist_tables = */ dist_tables.get(),
        /* num_queries = */ num_queries,
        /* codes = */ codes, /* ncodes = */ ncodes, /* heap = */ NULL,
        /* init_finalize_heap = */ init_finalize_heap);
  }

  inline uint32_t getNumSubquantizers() const { return _num_subquantizers; }

  inline uint32_t getNumBitsPerIndex() const { return _num_bits; }

  inline uint32_t getCentroidsCount() const { return _subq_centroids_count; }

  inline uint32_t getCodeSize() const { return _code_size; }

private:
  /**
   * @brief This function initializes a set of 2^(_num_bits) centroids spread
   around
   *  the mean of the input data in a hypercube configuration. For each
   dimension up
   *  to _num_bits, the centroid coordinates will be either mean[j] - maxm or
   *  mean[j] + maxm, creating a uniform distribution in a binary hypercube
   pattern.
   *
   *  The remaining coordinates (from _num_bits to dim) will be set to the mean
   of the
   *  dataset, collapsing the hypercube into a lower-dimensional hyperplane if
   _num_bits < dim.
   *  This kind of setup is a good initialization step for quantization or
   clustering
   *  as it ensures a good spread of centroids across the data space.
   *
   *
   * @param dim        This is essentially the subvector dimension (i.e.,
   *                    _subvector_dim)
   * @param num_bits   The number of bits we use to represent each pq code/index
   * @param n          The number of data points in the dataset
   * @param data       The actual dataset
   * @param centroids  Uninitialized centroids.
   *
   * TODO: Move the initialization to the CentroidsGenerator class

   */

  void initHypercube(uint32_t dim, uint32_t num_bits, uint64_t n,
                     const float *data, const float *centroids) {
    std::vector<float> means(dim);
    for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
      for (uint32_t dim_index = 0; dim_index < dim; dim_index++) {
        means[dim_index] += data[(vec_index * dim) + dim_index];
      }
    }

    float maxm = 0;
    for (uint32_t dim_index = 0; dim_index < dim; dim_index++) {
      means[dim_index] /= n;

      maxm = fabs(means[dim_index]) > maxm ? fabs(means[dim_index]) : maxm;
    }

    for (uint64_t i = 0; i < (1 << _num_bits); i++) {
      float *centroid = const_cast<float *>(centroids + (i * dim));
      for (uint64_t j = 0; j < _num_bits; j++) {
        centroid[j] = means[j] + (((i >> j) & 1) ? 1 : -1) * maxm;
      }

      for (uint64_t j = _num_bits; j < dim; j++) {
        centroid[j] = means[j];
      }
    }
  }

  /**
   * @brief Similar to `initHypercube` except that it pre-processes the input
   * data by computing Principal Component Analysis (PCA)
   *
   */
  void initHypercubePCA(uint32_t dim, uint32_t num_bits, uint64_t n,
                        const float *data, const float *centroids);

  // Bytes per indexed vector
  uint32_t _code_size;

  // This is typically called M in the literature, but we will
  // name it differently to differentiate M from HNSW (max edges per node)
  // and M in product quantization (number of subquantizers)
  uint32_t _num_subquantizers;

  // Number of bits per quantization index.
  uint32_t _num_bits;

  // The dimension of each subvector
  uint32_t _subvector_dim;

  // The number of centroids per subquantizer.
  // NOTE: If `_num_bits` = k, we can have maximum of 2^k
  // `_subq_centroids_count`. This simply follows from the fact that each
  // computed index can only index in [0, 2^k - 1] range.
  uint32_t _subq_centroids_count;

  // Centroid table. It will have size
  // (_num_subquantizers x _subq_centroids_count x _subvector_dim)
  std::vector<float> _centroids;

  // Represents centroids in a transposed form. This is useful while performing
  // the Asymmetric Distance Computation (ADC) where we are able to use the
  // transposed centroids to leverage SIMD instructions.
  // Layout: (_subvector_dim x _num_subquantizers x _subq_centroids_count)
  std::vector<float> _transposed_centroids;

  // Initialization
  enum TrainType {
    DEFAULT,
    HOT_START,     // The centroids are already initialized
    SHARED,        // Share dictionary across PQ segments
    HYPERCUBE,     // Initialize centroids with nbits-D hypercube
    HYPERCUBE_PCA, // Initialize centroids with nbits-D hypercube post PCA
                   // pre-processing. For now, this is not implemented. FAISS
                   // seems to believe that this is a good initialization, so we
                   // might test it out to see if it actually works well.
  };

  TrainType _train_type;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
};

} // namespace flatnav::quantization