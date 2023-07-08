

#pragma once

#include <algorithm>
#include <cstdint>
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
  CentroidsGenerator(uint32_t dim, uint32_t num_centroids,
                     uint32_t num_iterations = 25,
                     uint32_t max_points_per_centroid = 256,
                     bool normalized = true, bool verbose = false)
      : _dim(dim), _num_centroids(num_centroids),
        _clustering_iterations(num_iterations),
        _max_points_per_centroid(max_points_per_centroid),
        _normalized(normalized), _verbose(verbose),
        _centroids_initialized(false), _seed(3333) {}

  void initializeCentroids(const std::vector<std::vector<float>> &data,
                           const std::string &initialization_type) {
    // TODO: Move hypercube initialization from the ProductQuantizer class
    // to here.
    auto type = initialization_type;
    std::transform(type.begin(), type.end(), type.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (initialization_type == "default") {
      std::vector<uint64_t> indices(data.size());

      std::iota(indices.begin(), indices.end(), 0);
      std::mt19937 generator(_seed);
      std::vector<uint64_t> sample_indices(_num_centroids);
      std::sample(indices.begin(), indices.end(), sample_indices.begin(),
                  _num_centroids, generator);

      for (uint64_t i = 0; i < _num_centroids; i++) {
        auto sample_index = sample_indices[i];

        for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
          _centroids[(i * _dim) + dim_index] = data[sample_index][dim_index];
        }
      }
      _centroids_initialized = true;
      return;
    }

    throw std::invalid_argument(
        "Invalid centroids initialization initialization type: " +
        initialization_type);
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
   * @param vec_weights   weight associated with each datapoint: NULL or size n
   * @param n             The number of datapoints
   */
  void generateCentroids(const float *vectors, const float *vec_weights,
                         uint64_t n) {
    if (n < _num_centroids) {
      throw std::runtime_error(
          "Invalid configuration. The number of centroids " +
          std::to_string(_num_centroids) +
          " is bigger than the number of data points " + std::to_string(n));
    }

    std::vector<std::vector<float>> data(n, std::vector<float>(_dim));

    for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
      for (uint64_t dim_index = 0; dim_index < _dim; dim_index++) {
        data[vec_index][dim_index] = vectors[(vec_index * _dim) + dim_index];
      }
    }

    // Initialize the centroids by randomly sampling k centroids among the n
    // data points
    if (!_centroids_initialized) {
      _centroids.resize(_num_centroids * _dim);
      initializeCentroids(/* data = */ data,
                          /* initialization_type = */ "default");
    }

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
          float distance = 0.0;

          for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
            distance += (data[vec_index][dim_index] -
                         _centroids[(c_index * _dim) + dim_index]) *
                        (data[vec_index][dim_index] -
                         _centroids[(c_index * _dim) + dim_index]);
          }
          if (distance < min_distance) {
            assignment[vec_index] = c_index;
            min_distance = distance;
          }
        }
      }

      // Step 2: Update each centroid to be the mean of the assigned points
      std::vector<float> sums(_num_centroids * _dim, 0.0);
      std::vector<uint64_t> counts(_num_centroids, 0);

#pragma omp parallel for
      for (uint64_t vec_index = 0; vec_index < n; vec_index++) {
        for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
          sums[assignment[vec_index] * _dim + dim_index] +=
              data[vec_index][dim_index];
        }
        counts[assignment[vec_index]]++;
      }
#pragma omp parallel for
      for (uint32_t c_index = 0; c_index < _num_centroids; c_index++) {
        for (uint32_t dim_index = 0; dim_index < _dim; dim_index++) {
          _centroids[c_index * _dim + dim_index] =
              counts[c_index]
                  ? sums[c_index * _dim + dim_index] / counts[c_index]
                  : 0;
        }
      }
    }
  }

  inline std::vector<float> &centroids() { return _centroids; }

private:
  uint32_t _dim;

  // Number of cluster centroids
  uint32_t _num_centroids;
  // Centroids. This will be an array of k * _dim floats
  // where k is the number of centroids
  std::vector<float> _centroids;

  // Number of clustering iterations
  uint32_t _clustering_iterations;
  // normalize centroids if set to true

  // limit the dataset size. If the number of datapoints
  // exceeds k * _max_points_per_centroid, we use subsampling
  uint32_t _max_points_per_centroid;
  bool _normalized;
  bool _verbose;
  bool _centroids_initialized;

  // seed for random number generator;
  int _seed;
};

} // namespace flatnav::quantization