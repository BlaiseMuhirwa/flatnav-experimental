#pragma once

#include <algorithm>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace flatnav::quantization {

class CentroidsGenerator {
public:
  /**
   * @brief Construct a new Centroids Generator object
   *
   * @param dim                       The dimension of the input vectors
   * @param num_centroids             The number of centroids to generate
   * @param num_iterations            The number of clustering iterations. From
   *                                  experimentation, clustering iterations >
   * 60 does not improve the quality of the centroids regardless of the
   * initialization strategy. As one would expect, `kmeans++` initialization
   * requires less clustering initialization to converge to a good solution than
   * `default` initialization.
   * @param normalized                Whether to normalize the centroids
   * @param verbose                   Whether to print verbose output
   * @param seed                      The seed for the random number generator
   */
  CentroidsGenerator(uint32_t dim, uint32_t num_centroids,
                     uint32_t num_iterations = 62, bool normalized = true,
                     bool verbose = false, int seed = 3333)
      : _dim(dim), _num_centroids(num_centroids),
        _clustering_iterations(num_iterations), _normalized(normalized),
        _verbose(verbose), _centroids_initialized(false), _seed(seed),
        _initialization_type("default") {}

  void initializeCentroids(
      const float *data, uint64_t n,
      const std::function<float(const float *, const float *)> &distance_func) {
    auto initialization_type = _initialization_type;
    std::transform(initialization_type.begin(), initialization_type.end(),
                   initialization_type.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (_centroids.size() != _num_centroids * _dim) {
      _centroids.resize(_num_centroids * _dim);
    }

    if (initialization_type == "default") {
      randomInitialize(data, n);
    } else if (initialization_type == "kmeans++") {
      kmeansPlusPlusInitialize(data, n, distance_func);
    } else if (initialization_type == "hypercube") {
      hypercubeInitialize(data, n);
    } else {
      throw std::invalid_argument(
          "Invalid centroids initialization initialization type: " +
          initialization_type);
    }
    _centroids_initialized = true;
  }

  /**
   * @brief Run k-means clustering algorithm to compute D-dimensional
   * centroids given n D-dimensional vectors.

   * The algorithm proceeds as follows:
   * - Select k datapoints as the initial centroids (we use a random
   initialization for now).
   *   We can also use another strategy such as kmeans++ initialization due to
   Arthur and
   *   Vassilvitskii.
   https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
   *
   * - Assign each data point to its nearest centroid based on l2 distances.
   * - Update each centroid to be the mean of its assigned datapoints
   * - Repeat steps 2 & 3 until we reach _num_iterations
   *
   * @param vectors       the input datapoints
   * @param vec_weights   weight associated with each datapoint: NULL or size n.
   This is adapted from FAISS, but it is not currently used.
   * @param n             The number of datapoints
   * @param distance_func The distance function to use (e.g. l2 distance or
   cosinde/inner product)
   */
  void generateCentroids(
      const float *vectors, const float *vec_weights, uint64_t n,
      const std::function<float(const float *, const float *)> &distance_func) {
    if (n < _num_centroids) {
      throw std::runtime_error(
          "Invalid configuration. The number of centroids: " +
          std::to_string(_num_centroids) +
          " is bigger than the number of data points: " + std::to_string(n));
    }

    initializeCentroids(vectors, n, distance_func);

    // Temporary array to store assigned centroids for each vector
    std::vector<uint32_t> assignment(n);

    // K-means loop
    for (uint32_t iteration = 0; iteration < _clustering_iterations;
         iteration++) {
// Step 1. Find the minimizing centroid based on l2 distance
#pragma omp parallel for
      for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
        float min_distance = std::numeric_limits<float>::max();

        for (uint32_t c_index = 0; c_index < _num_centroids; c_index++) {
          // Get distance using the distance function
          float *vector = const_cast<float *>(vectors + (vec_index * _dim));
          float *centroid =
              const_cast<float *>(_centroids.data() + (c_index * _dim));
          auto distance = distance_func(vector, centroid);

          if (distance < min_distance) {
            assignment[vec_index] = c_index;
            min_distance = distance;
          }
        }
      }

      // Step 2: Update each centroid to be the mean of the assigned points
      std::vector<double> sums(_num_centroids * _dim, 0.0);
      std::vector<uint64_t> counts(_num_centroids, 0);

#pragma omp parallel for
      for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
        for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
#pragma omp atomic
          sums[assignment[vec_index] * _dim + dim_index] +=
              vectors[vec_index * _dim + dim_index];
        }
#pragma omp atomic
        counts[assignment[vec_index]]++;
      }
#pragma omp parallel for
      for (uint32_t c_index = 0; c_index < _num_centroids; c_index++) {
        for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
          _centroids[c_index * _dim + dim_index] =
              counts[c_index]
                  ? sums[c_index * _dim + dim_index] / counts[c_index]
                  : _centroids[c_index * _dim + dim_index];
        }
      }
    }
  }

  inline const float *centroids() const { return _centroids.data(); }

  inline void setInitializationType(const std::string &initialization_type) {
    _initialization_type = initialization_type;
  }

private:
  /**
   * @brief Initialize the centroids by randomly sampling k centroids among the
   * n data points
   * @param data  The input data points
   * @param n     The number of data points
   */
  void randomInitialize(const float *data, uint64_t n) {
    std::vector<uint64_t> indices(n);

    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 generator(_seed + 1);
    std::vector<uint64_t> sample_indices(_num_centroids);
    std::sample(indices.begin(), indices.end(), sample_indices.begin(),
                _num_centroids, generator);

    for (uint32_t i = 0; i < _num_centroids; i++) {
      auto sample_index = sample_indices[i];

      for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
        _centroids[(i * _dim) + dim_index] =
            data[(sample_index * _dim) + dim_index];
      }
    }
  }

  /**
   * @brief Initialize the centroids using the kmeans++ algorithm.
   * The algorithm proceeds as follows:
   * - Select the first centroid at random
   * - For k-1 remaining centroids, do
   *  - For each point in the dataset, compute the squared distance to the
   * nearest centroid that has already been chosen.
   *  - Choose the next centroid from the dataset with probability proportional
   * to the squared distance to the nearest centroid. This means that points
   * that are farther from existing centroids are more likely to be selected as
   * the next centroids.
   *
   * @param data  The input data points
   * @param n     The number of data points
   */
  void kmeansPlusPlusInitialize(
      const float *data, uint64_t n,
      const std::function<float(const float *, const float *)> &distance_func) {
    std::mt19937 generator(_seed);
    std::uniform_int_distribution<uint64_t> distribution(0, n - 1);

    // Step 1. Select the first centroid at random
    uint64_t first_centroid_index = distribution(generator);
    for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
      _centroids[dim_index] = data[first_centroid_index * _dim + dim_index];
    }

    std::vector<double> min_squared_distances(
        n, std::numeric_limits<double>::max());

    // Step 2. For k-1 remaining centroids
    for (uint32_t cent_idx = 1; cent_idx < _num_centroids; cent_idx++) {
      // Compute squared distances from the points to the nearest centroid
      double sum = 0.0;

#pragma omp parallel for reduction(+ : sum)
      for (uint64_t i = 0; i < n; i++) {
        double min_distance = std::numeric_limits<double>::max();

        for (uint64_t c = 0; c < cent_idx; c++) {

          float *centroid = const_cast<float *>(_centroids.data() + (c * _dim));
          float *vector = const_cast<float *>(data + (i * _dim));
          auto distance = distance_func(centroid, vector);

          if (distance < min_distance) {
            min_distance = distance;
          }
        }
        min_squared_distances[i] = min_distance;
        sum += min_distance;
      }

      // Choose the next centroid based on weighted probability
      std::uniform_real_distribution<double> distribution(0.0, sum);
      double threshold = distribution(generator);
      sum = 0.0;
      uint64_t next_centroid_index = 0;
      for (; next_centroid_index < n; next_centroid_index++) {
        sum += min_squared_distances[next_centroid_index];
        if (sum >= threshold) {
          break;
        }
      }

      // Add selected centroid the the centroids array
      for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
        _centroids[cent_idx * _dim + dim_index] =
            data[next_centroid_index * _dim + dim_index];
      }
    }
  }

  /**
 * Adapted from FAISS
 *
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
 * @param data       The actual dataset
 * @param n          The number of data points in the dataset
 *

 */

  void hypercubeInitialize(const float *data, uint64_t n) {

    std::vector<float> means(_dim);
    for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
      for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
        means[dim_index] += data[(vec_index * _dim) + dim_index];
      }
    }

    float maxm = 0;
    for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
      means[dim_index] /= n;

      maxm = fabs(means[dim_index]) > maxm ? fabs(means[dim_index]) : maxm;
    }

    float *centroids = _centroids.data();
    auto num_bits = log2(_num_centroids);

    for (uint32_t i = 0; i < _num_centroids; i++) {
      float *centroid = const_cast<float *>(centroids + (i * _dim));
      for (uint32_t j = 0; j < num_bits; j++) {
        centroid[j] = means[j] + (((i >> j) & 1) ? 1 : -1) * maxm;
      }

      for (uint32_t j = num_bits; j < _dim; j++) {
        centroid[j] = means[j];
      }
    }
  }

  uint32_t _dim;

  // Number of cluster centroids
  uint32_t _num_centroids;
  // Centroids. This will be an array of k * _dim floats
  // where k is the number of centroids
  std::vector<float> _centroids;

  // Number of clustering iterations
  uint32_t _clustering_iterations;
  // normalize centroids if set to true

  bool _normalized;
  bool _verbose;
  bool _centroids_initialized;

  // seed for random number generator;
  int _seed;

  std::string _initialization_type;

  friend class cereal::access;
  template <typename Archive> void serialize(Archive &ar) {
    ar(_dim, _num_centroids, _centroids, _clustering_iterations, _normalized,
       _verbose, _centroids_initialized, _seed, _initialization_type);
  }
};

} // namespace flatnav::quantization