#pragma once

#include <flatnav/util/simd_base_types.h>

namespace flatnav::util {

#if defined(USE_AVX512)

static float computeIP_Avx512(const void *x, const void *y,
                              const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  // Align to 16-floats boundary
  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd16float32 product, v1, v2;

  simd16float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 16;
    pointer_y += 16;
  }
  float total = sum.reduce_add();
  return 1.0f - total;
}

#endif // USE_AVX512

#if defined(USE_AVX)
static float computeIP_Avx(const void *x, const void *y,
                           const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd8float32 product, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 8;
    pointer_y += 8;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    product = v1 * v2;
    sum += product;
    pointer_x += 8;
    pointer_y += 8;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

static float computeIP_Avx_4aligned(const void *x, const void *y,
                                    const size_t &dimension) {

  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *first_chunk_end = pointer_x + (dimension >> 4 << 4);
  const float *second_chunk_end = pointer_x + (dimension >> 2 << 2);

  simd8float32 v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != first_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 8;
    pointer_y += 8;
  }

  // TODO: See if we can reduce this to fewer instructions
  simd4float32 aggregate =
      simd4float32(sum.get_low()) + simd4float32(sum.get_high());
  simd4float32 v1_residual, v2_residual;

  while (pointer_x != second_chunk_end) {
    v1_residual.loadu(pointer_x);
    v2_residual.loadu(pointer_y);
    aggregate += (v1_residual * v2_residual);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = aggregate.reduce_add();
  return 1.0f - total;
}

#endif // USE_AVX

#if defined(USE_SSE)

const float computeIP_Sse(const void *x, const void *y,
                          const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd4float32 v1, v2;
  simd4float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

const float computeIP_Sse_4aligned(const void *x, const void *y,
                                   const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));
  const float *first_chunk_end = pointer_x + (dimension >> 4 << 4);
  const float *second_chunk_end = pointer_x + (dimension >> 2 << 2);

  simd4float32 v1, v2;
  simd4float32 sum(0.0f);
  while (pointer_x != first_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;

    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  while (pointer_x != second_chunk_end) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    sum += (v1 * v2);
    pointer_x += 4;
    pointer_y += 4;
  }

  float total = sum.reduce_add();
  return 1.0f - total;
}

const float computeIP_SseWithResidual_16(const void *x, const void *y,
                                         const size_t &dimension) {
  size_t aligned_dimension = dimension >> 4 << 4;
  size_t residual_dimension = dimension - aligned_dimension;

  // We need to subtract 1.0f from the result, and then multiply by -1.0f
  // in order to get the actual dot product.
  float first_chunk_sum = computeIP_Sse(x, y, aligned_dimension);
  first_chunk_sum -= 1.0f;
  first_chunk_sum *= -1.0f;

  float residual_sum = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + aligned_dimension;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + aligned_dimension;
  for (size_t i = 0; i < residual_dimension; i++) {
    residual_sum += pointer_x[i] * pointer_y[i];
  }
  return 1.0f - (first_chunk_sum + residual_sum);
}

const float computeIP_SseWithResidual_4(const void *x, const void *y,
                                        const size_t &dimension) {
  size_t aligned_dimension = dimension >> 2 << 2;
  size_t residual_dimension = dimension - aligned_dimension;

  // We need to subtract 1.0f from the result, and then multiply by -1.0f
  // in order to get the actual dot product.
  float first_chunk_sum = computeIP_Sse_4aligned(x, y, aligned_dimension);
  first_chunk_sum -= 1.0f;
  first_chunk_sum *= -1.0f;

  float residual_sum = 0.0f;
  float *pointer_x =
      static_cast<float *>(const_cast<void *>(x)) + aligned_dimension;
  float *pointer_y =
      static_cast<float *>(const_cast<void *>(y)) + aligned_dimension;
  for (size_t i = 0; i < residual_dimension; i++) {
    residual_sum += pointer_x[i] * pointer_y[i];
  }
  return 1.0f - (first_chunk_sum + residual_sum);
}

#endif // USE_SSE

} // namespace flatnav::util
