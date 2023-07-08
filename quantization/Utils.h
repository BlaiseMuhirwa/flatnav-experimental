
#pragma once

#include <cstdint>
#include <limits>

#ifdef __SSE2__
#include <immintrin.h>
#endif

// #ifndef __aarch64__
// #include <arm_neon.h>
// #endif

namespace flatnav {

#ifdef __AVX__
#define USE_AVX
#endif

/**
 * @brief Return the squared L2 distance between two vectors of the specified
 * dimension. Checks for dimensionality are assumed to have occured prior to
 * the invocation of this function.
 *
 * @param x
 * @param y
 * @param dim
 * @return squared L2 distance
 */
static float squaredL2(const float *x, const float *y, uint32_t dim) {
  float l2_squared = 0;
  for (uint32_t i = 0; i < dim; i++) {
    l2_squared += (x[i] - y[i]) * (x[i] - y[i]);
  }
  return l2_squared;
}

/**
 * @brief Stores all squared L2 distances between the input vector x and all
 * other vector(s) y.
 *
 * @param distances_buffer
 * @param x
 * @param y
 * @param dim
 * @param target_set_size
 */
static void registerSquaredL2Distances(float *distances_buffer, const float *x,
                                       const float *y, uint32_t dim,
                                       uint64_t target_set_size) {
  for (uint64_t i = 0; i < target_set_size; i++) {
    distances_buffer[i] = squaredL2(x, y, dim);
    y += dim;
  }
}

static uint64_t squaredL2WithKNeighbors_(float *distances_buffer,
                                         const float *x, const float *y,
                                         uint32_t dim,
                                         uint64_t target_set_size) {
  registerSquaredL2Distances(distances_buffer, x, y, dim, target_set_size);
  uint64_t minimizer = 0;
  float minimum_distance = std::numeric_limits<float>::max();

  for (uint64_t i = 0; i < target_set_size; i++) {
    if (distances_buffer[i] < minimum_distance) {
      minimum_distance = distances_buffer[i];
      minimizer = i;
    }
  }
  return minimizer;
}

/** Compute target_set_size square L2 distances between x and a set f contiguous
 * y vectors and return the index of the nearest neighbor.
 *
 * @param distances_buffer       buffer storing distances computed
 * @param x                          vector for which the distances are to be
 * computed
 * @param y                          pointer to the starting target vector
 * @param dim                        the dimension of the vectors
 * @param target_set_size            the size of the contiguous y vectors
 *
 * @return 0 if target_set_size equals 0. Otherwise, the index of the
 * nearest vector.
 */
static uint64_t squaredL2WithKNeighbors(float *distances_buffer, const float *x,
                                        const float *y, uint32_t dim,
                                        uint64_t target_set_size) {

  if (target_set_size == 0) {
    return 0;
  }
  return squaredL2WithKNeighbors_(distances_buffer, x, y, dim, target_set_size);
}

} // namespace flatnav