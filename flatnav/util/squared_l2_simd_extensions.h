#pragma once

#include <flatnav/util/simd_base_types.h>

namespace flatnav::util {

#if defined(USE_AVX512)
static float computeL2_Avx512(const void *x, const void *y,
                              const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  // Align to 16-floats boundary
  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd16float32 difference, v1, v2;

  simd16float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 16;
    pointer_y += 16;
  }
  return sum.reduce_add();
}

#endif // USE_AVX512

#if defined(USE_AVX)

static float computeL2_Avx2(const void *x, const void *y,
                            const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension & ~7);
  simd8float32 difference, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;

    pointer_x += 8;
    pointer_y += 8;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;

    pointer_x += 8;
    pointer_y += 8;
  }

  float result[8];
  sum.storeu(result);
  return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] +
         result[6] + result[7];
}

#endif // USE_AVX

#if defined(USE_SSE)

static float computeL2_Sse(const void *x, const void *y,
                           const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

static float computeL2_Sse4Aligned(const void *x, const void *y,
                                   const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 2 << 2);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

static float computeL2_SseWithResidual_16(const void *x, const void *y,
                                          const size_t &dimension) {

  size_t dimension_aligned = dimension >> 4 << 4;
  float aligned_distance = computeL2_Sse(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + dimension_aligned;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

static float computeL2_Sse4aligned(const void *x, const void *y,
                                   const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 2 << 2);
  simd4float32 difference, v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 4;
    pointer_y += 4;
  }

  return sum.reduce_add();
}

static float computeL2_SseWithResidual_4(const void *x, const void *y,
                                         const size_t &dimension) {
  size_t dimension_aligned = dimension >> 2 << 2;
  float aligned_distance = computeL2_Sse4aligned(x, y, dimension_aligned);
  size_t residual_dimension = dimension - dimension_aligned;
  float residual_distance = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + dimension_aligned;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + dimension_aligned;
  for (size_t i = 0; i < residual_dimension; i++) {
    float difference = *pointer_x - *pointer_y;
    residual_distance += difference * difference;
    pointer_x++;
    pointer_y++;
  }
  return aligned_distance + residual_distance;
}

#endif // USE_SSE

} // namespace flatnav::util