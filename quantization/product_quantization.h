#pragma once

#include <algorithm>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <flatnav/distance_interface.h>
#include <flatnav/distances/inner_product_distance.h>
#include <flatnav/distances/squared_l2_distance.h>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <quantization/CentroidsGenerator.h>
#include <quantization/Utils.h>
#include <queue>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace flatnav::quantization {

using flatnav::METRIC_TYPE;
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

class ProductQuantizer : public flatnav::DistanceInterface<ProductQuantizer> {
  friend class flatnav::DistanceInterface<ProductQuantizer>;

  // Represents the block size used in ProductQuantizer::computePQCodes
  static const uint64_t BLOCK_SIZE = 256 * 1024;

public:
  // Constructor for serializaiton
  ProductQuantizer() = default;

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
  ProductQuantizer(uint32_t dim, uint32_t M, uint32_t nbits,
                   METRIC_TYPE metric_type)
      : _num_subquantizers(M), _num_bits(nbits), _is_trained(false),
        _metric_type(metric_type), _train_type(TrainType::DEFAULT) {

    if (dim % _num_subquantizers) {
      throw std::invalid_argument("The dataset dimension must be a multiple of "
                                  "the desired number of sub-quantizers.");
    }
    _code_size = (_num_bits * 8 + 7) / 8;
    _subvector_dim = dim / _num_subquantizers;

    if (_metric_type == METRIC_TYPE::EUCLIDEAN) {
      _distance = SquaredL2Distance(_subvector_dim);
    } else if (_metric_type == METRIC_TYPE::INNER_PRODUCT) {
      _distance = InnerProductDistance(_subvector_dim);
    } else {
      throw std::invalid_argument("Invalid metric type");
    }

    _subq_centroids_count = 1 << _num_bits;
    _centroids.resize(_subq_centroids_count * dim);

    _dist_func = getDistFuncFromVariant();
  }

  // Return a pointer to the centroids associated with a given subvector
  const float *getCentroids(uint32_t subvector_index, uint32_t i) const {
    auto index = (subvector_index * _subq_centroids_count + i) * _subvector_dim;
    return &_centroids[index];
  }

  void setParameters(const float *centroids_, int m) {
    float *centroids = const_cast<float *>(getCentroids(m, 0));
    auto bytes_to_copy = _subq_centroids_count * _subvector_dim * sizeof(float);

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

    PQCodeManager<uint8_t> code_manager(/* code = */ code,
                                        /* nbits = */ 8);

    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      const float *subvector = vector + (m * _subvector_dim);
      uint64_t minimizer_index = flatnav::distanceWithKNeighbors(
          /* distances_tmp_buffer = */ distances.data(), /* x = */ subvector,
          /* y = */ getCentroids(m, 0), /* dim = */ _subvector_dim,
          /* target_set_size = */ _subq_centroids_count,
          /* dist_func = */ _dist_func);

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
          /* n = */ n * _num_subquantizers,
          /* distance_func = */ _dist_func);

      for (uint32_t m = 0; m < _num_subquantizers; m++) {
        setParameters(/* centroids_ = */ centroids_generator.centroids(),
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

    float *slice = new float[n * _subvector_dim];
    auto dim = _subvector_dim * _num_subquantizers;

    // Arrange the vectors such that the first subvector of each vector is
    // contiguous, then the second subvector, and so on.
    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
        std::memcpy(slice + (vec_index * _subvector_dim),
                    vectors + (vec_index * dim) + (m * _subvector_dim),
                    _subvector_dim * sizeof(float));
      }

      switch (final_train_type) {
      case TrainType::HYPERCUBE:
        centroids_generator.setInitializationType("hypercube");
        break;

      case TrainType::HOT_START:
        std::memcpy((void *)centroids_generator.centroids(), getCentroids(m, 0),
                    _subvector_dim * _subq_centroids_count * sizeof(float));
        break;

      default:;
      }

      // generate the actual centroids
      centroids_generator.generateCentroids(
          /* vectors = */ slice, /* vec_weights = */ NULL, /* n = */ n,
          /* distance_func = */ _dist_func);

      setParameters(/* centroids_ = */ centroids_generator.centroids(),
                    /* m = */ m);
    }

    _is_trained = true;
    computeSymmetricDistanceTables();
    delete[] slice;
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
        /* code = */ const_cast<uint8_t *>(code),
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
   * @param dist_table output table, size (_num_subquantizers x
   * _subq_centroids_count)
   */
  void
  computeDistanceTable(const float *vector, float *dist_table,
                       const std::function<float(const float *, const float *)>
                           &dist_func) const {

    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      flatnav::copyDistancesIntoBuffer(
          /* distances_buffer = */ dist_table + (m * _subq_centroids_count),
          /* x = */ vector + (m * _subvector_dim), /* y = */ getCentroids(m, 0),
          /* dim = */ _subvector_dim,
          /* target_set_size = */ _subq_centroids_count,
          /* dist_func = */ dist_func);
    }
  }

  void computeDistanceTables(const float *vectors, float *dist_tables,
                             uint64_t n) const {

    // TODO: Use SIMD
    auto dim = _subvector_dim * _num_subquantizers;
#pragma omp parallel for if (n > 1)
    for (uint64_t i = 0; i < n; i++) {
      computeDistanceTable(vectors + (i * dim),
                           dist_tables +
                               (i * _subq_centroids_count * _num_subquantizers),
                           _dist_func);
    }
  }

  ////////////////////////////////////////////////////////////////////////////////////////
  //                     Implementation of DistanceInterface Methods //
  //                                                                                    //
  ////////////////////////////////////////////////////////////////////////////////////////
  inline size_t getDimension() const { return getCodeSize(); }

  inline size_t dataSizeImpl() { return getCodeSize(); }

  void transformDataImpl(void *destination, const void *src) {
    uint8_t *code = new uint8_t[_code_size]();
    computePQCode(static_cast<const float *>(src), code);

    std::memcpy(destination, code, _code_size);

    delete[] code;
  }

  /**
   * @brief Computes the distance between a query vector and a database vector.
   * NOTE: The first vector is expected to the query vector and the second one
   * a database vector.
   *
   * @param x         query vector
   * @param y         database vector
   * @return
   */
  float asymmetricDistanceImpl(const void *x, const void *y) const {
    assert(_is_trained);

    float *x_ptr = (float *)(x);
    uint8_t *y_ptr = (uint8_t *)(y);

    float *dist_table = new float[_subq_centroids_count * _num_subquantizers];

    computeDistanceTable(/* vector = */ x_ptr,
                         /* dist_table = */ dist_table,
                         /* dist_func = */ _dist_func);

    float distance = 0.0;
    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      distance += dist_table[(m * _subq_centroids_count) + y_ptr[m]];
    }
    delete[] dist_table;
    return distance;
  }

  /**
   * @brief Computes the distance between two database vectors by utilizing
   * the pre-computed distance tables.
   *
   * @param x
   * @param y
   * @return
   */
  float symmetricDistanceImpl(const void *x, const void *y) const {
    assert(_is_trained);

    uint8_t *code1 = (uint8_t *)(x);
    uint8_t *code2 = (uint8_t *)(y);

    float distance = 0.0;

    // Get a pointer to the distance table for the first subquantizer
    const float *dist_table = _symmetric_distance_tables.data();

    for (uint32_t m = 0; m < _num_subquantizers; m++) {
      distance += dist_table[(code1[m] * _subq_centroids_count) + code2[m]];
      dist_table += _subq_centroids_count * _subq_centroids_count;
    }
    return distance;
  }

  float distanceImpl(const void *x, const void *y, bool asymmetric) const {
    if (asymmetric) {
      return asymmetricDistanceImpl(x, y);
    }
    return symmetricDistanceImpl(x, y);
  }

  void getSummaryImpl() const {
    std::cout << "\nProduct Quantizer Parameters" << std::flush;
    std::cout << "-----------------------------" << std::flush;
    std::cout << "Number of subquantizers (M): " << _num_subquantizers << "\n"
              << std::flush;
    std::cout << "Number of bits per index: " << _num_bits << "\n"
              << std::flush;
    std::cout << "Subvector dimension: " << _subvector_dim << "\n"
              << std::flush;
    std::cout << "Subquantizer centroids count: " << _subq_centroids_count
              << "\n"
              << std::flush;
    std::cout << "Code size: " << _code_size << "\n" << std::flush;
    std::cout << "Is trained: " << _is_trained << "\n" << std::flush;
    std::cout << "Train type: " << _train_type << "\n" << std::flush;
  }

  inline uint32_t getNumSubquantizers() const { return _num_subquantizers; }

  inline uint32_t getNumBitsPerIndex() const { return _num_bits; }

  inline uint32_t getCentroidsCount() const { return _subq_centroids_count; }

  inline uint32_t getCodeSize() const { return _code_size; }

  inline bool isTrained() const { return _is_trained; }

private:
  // NOTE: This is a hack to get around the fact that the PQ class needs to know
  // which distance function to use. So, this function allows us to just extract
  // the distance function pointer since that's the only thing we care about.
  // There's gotta be a cleaner way to not have to do this, but this will do for
  // now.

  std::function<float(const float *, const float *)>
  getDistFuncFromVariant() const {
    if (_distance.index() == 0) {
      return [local_distance = _distance](const float *a,
                                          const float *b) -> float {
        return std::get<SquaredL2Distance>(local_distance).distanceImpl(a, b);
      };
    }
    return [local_distance = _distance](const float *a,
                                        const float *b) -> float {
      return std::get<InnerProductDistance>(local_distance).distanceImpl(a, b);
    };
  }

  /**
   * @brief Computes pair-wise distances between all pairs of centroids
   * for each subquantizer.
   * The symmetric distance table is essentially a 3D array of size
   * (_num_subquantizers x _subq_centroids_count x _subq_centroids_count).
   * This means that for each subquantizer, we have a 2D symmetric matrix
   * of size (_subq_centroids_count x _subq_centroids_count).
   * The current implementation, in fact, is not efficient since we are
   * computing the distance between all pairs of centroids twice. We can
   * improve this by leveraging the symmetric property of each matrix.
   *
   * @TODO: Use BLAS in regimes where _subvector_dim is large since we can gain
   * some speed.
   *
   */
  void computeSymmetricDistanceTables() {
    _symmetric_distance_tables.resize(
        _num_subquantizers * _subq_centroids_count * _subq_centroids_count);

#pragma omp parallel for
    for (uint64_t mk = 0; mk < _num_subquantizers * _subq_centroids_count;
         mk++) {
      auto m = mk / _subq_centroids_count;
      auto k = mk % _subq_centroids_count;
      const float *centroids =
          _centroids.data() + (m * _subq_centroids_count * _subvector_dim);
      const float *centroid_k = centroids + (k * _subvector_dim);
      float *dist_table = _symmetric_distance_tables.data() +
                          (m * _subq_centroids_count * _subq_centroids_count);

      flatnav::copyDistancesIntoBuffer(
          /* distances_buffer = */ dist_table + (k * _subq_centroids_count),
          /* x = */ centroid_k, /* y = */ centroids,
          /* dim = */ _subvector_dim,
          /* target_set_size = */ _subq_centroids_count,
          /* dist_func = */ _dist_func);
    }
  }

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

  std::vector<float> _symmetric_distance_tables;

  // Indicates if the PQ has been trained or not
  bool _is_trained;

  METRIC_TYPE _metric_type;

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

  std::variant<SquaredL2Distance, InnerProductDistance> _distance;

  std::function<float(const float *, const float *)> _dist_func;

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &archive) {

    archive(_code_size, _num_subquantizers, _num_bits, _subvector_dim,
            _subq_centroids_count, _centroids, _symmetric_distance_tables,
            _is_trained, _metric_type, _train_type);

    if constexpr (Archive::is_loading::value) {
      // loading PQ
      if (_metric_type == METRIC_TYPE::EUCLIDEAN) {
        _distance = SquaredL2Distance(_subvector_dim);
      } else if (_metric_type == METRIC_TYPE::INNER_PRODUCT) {
        _distance = InnerProductDistance(_subvector_dim);
      } else {
        throw std::invalid_argument("Invalid metric type");
      }
      _dist_func = getDistFuncFromVariant();
    }
  }
};

} // namespace flatnav::quantization