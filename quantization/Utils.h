
#pragma once

#include <cstdint>
#include <flatnav/distances/inner_product_distance.h>
#include <flatnav/distances/squared_l2_distance.h>
#include <limits>
#include <variant>

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
 * @brief Stores all squared L2 distances between the input vector x and all
 * other vector(s) y.
 *
 * @param distances_buffer
 * @param x
 * @param y
 * @param dim
 * @param target_set_size
 */
static void copyDistancesIntoBuffer(
    float *distances_buffer, const float *x, const float *y, uint32_t dim,
    uint64_t target_set_size,
    const std::function<float(const float *, const float *)> &dist_func) {

  for (uint64_t i = 0; i < target_set_size; i++) {
    distances_buffer[i] = dist_func(x, y);
    y += dim;
  }
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
static uint64_t distanceWithKNeighbors(
    float *distances_buffer, const float *x, const float *y, uint32_t dim,
    uint64_t target_set_size,
    const std::function<float(const float *, const float *)> &dist_func) {

  if (target_set_size == 0) {
    return 0;
  }
  copyDistancesIntoBuffer(distances_buffer, x, y, dim, target_set_size,
                          dist_func);
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

} // namespace flatnav