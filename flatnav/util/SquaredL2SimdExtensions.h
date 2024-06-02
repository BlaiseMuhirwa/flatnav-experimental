#pragma once

#include <flatnav/util/SimdBaseTypes.h>

namespace flatnav::util {

// Explicitly expresses that narrowing is either acceptable or known impossible.
template <typename T, typename U> constexpr T narrow_cast(U&& u) noexcept {
    return static_cast<T>(std::forward<U>(u));
}


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

#if defined(USE_AVX512BW) && defined(USE_AVX512VNNI)

// template <size_t N> static inline __mmask32 create_mask(const size_t &length)
// {
//   __mmask32 mask = 0;
//   for (size_t i = 0; i < N; ++i) {
//     mask |= (i < length) ? (1UL << i) : 0;
//   }
//   return mask;
// }

constexpr __mmask32 create_mask(size_t remaining) {
  // If remaining is 32 or more, we want to load everything, so the mask is all
  // 1s. If remaining is less, shift a 1 up to the remaining bit, subtracting
  // one to get a mask with that many 1s.
  // return remaining >= 32 ? static_cast<__mmask32>(-1) : (1UL << remaining) -
  // 1;
  return (1UL << remaining) - 1;
}

template <size_t VecLength>
constexpr mask_intrinsic_from_length<VecLength>
create_mask(size_t dimension) {
  using MaskType = mask_repr_t<VecLength>;
  constexpr MaskType one{0x1};
  MaskType shift = dimension % VecLength;
  MaskType mask_raw =
      shift == 0 ? std::numeric_limits<MaskType>::max() : (one << shift) - one;
  return mask_raw;
}

template <size_t VecLength> constexpr mask_intrinsic_from_length<VecLength> no_mask() {
    return std::numeric_limits<mask_repr_t<VecLength>>::max();
}

static constexpr size_t div_round_up(size_t x, size_t y) {
  return (x / y) + static_cast<size_t>((x % y) != 0);
}

template <size_t Step> static constexpr bool islast(size_t N, size_t i) {
  size_t last_iter = Step * (div_round_up(N, Step) - 1);
  return i == last_iter;
}

static float compute(const int8_t *a, const int8_t *b, const size_t &length) {
  auto sum = _mm512_setzero_epi32();
  size_t j = 0;

  auto mask = create_mask<32>(length);
  auto all = no_mask<32>();

  for (; j < length; j += 32) {
    auto temp_a = _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, a + j);
    auto va = _mm512_cvtepi8_epi16(temp_a);

    auto temp_b = _mm256_maskz_loadu_epi8(islast<32>(length, j) ? mask : all, b + j);
    auto vb = _mm512_cvtepi8_epi16(temp_b);

    auto diff = _mm512_sub_epi16(va, vb);
    sum = _mm512_dpwssd_epi32(sum, diff, diff);
  }
  return narrow_cast<float>(_mm512_reduce_add_epi32(sum));
}

static float computeL2_Avx512_int8(const void *x, const void *y,
                                   const size_t &dimension) {
  int8_t *pointer_x = static_cast<int8_t *>(const_cast<void *>(x));
  int8_t *pointer_y = static_cast<int8_t *>(const_cast<void *>(y));

  return flatnav::util::compute(pointer_x, pointer_y, dimension);
}

#endif // USE_AVX512BW && USE_AVX512VNNI

#endif // USE_AVX512

#if defined(USE_AVX)

static float computeL2_Avx2(const void *x, const void *y,
                            const size_t &dimension) {
  float *pointer_x = static_cast<float *>(const_cast<void *>(x));
  float *pointer_y = static_cast<float *>(const_cast<void *>(y));

  const float *end_x = pointer_x + (dimension >> 4 << 4);
  simd8float32 difference, v1, v2;
  simd8float32 sum(0.0f);

  while (pointer_x != end_x) {
    v1.loadu(pointer_x);
    v2.loadu(pointer_y);
    difference = v1 - v2;
    sum += difference * difference;
    pointer_x += 8;
    pointer_y += 8;
  }

  return sum.reduce_add();
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

// This function computes the L2 distance between two int8 vectors using SSE2
// instructions.
static float computeL2_Sse_int8(const void *x, const void *y,
                                const size_t &dimension) {
  int8_t *pointer_x = static_cast<int8_t *>(const_cast<void *>(x));
  int8_t *pointer_y = static_cast<int8_t *>(const_cast<void *>(y));

  __m128i sum = _mm_setzero_si128();
  size_t aligned_dimension = dimension & ~0xF;
  size_t i = 0;

  for (; i < aligned_dimension; i += 16) {
    __m128i vx = _mm_loadu_si128(reinterpret_cast<__m128i *>(pointer_x + i));
    __m128i vy = _mm_loadu_si128(reinterpret_cast<__m128i *>(pointer_y + i));
    __m128i diff = _mm_sub_epi8(vx, vy);

    // Convert to 16-bit and square
    __m128i diff_squared =
        _mm_madd_epi16(_mm_cvtepi8_epi16(diff), _mm_cvtepi8_epi16(diff));

    // Accumulate in 32-bit integer
    sum = _mm_add_epi32(sum, diff_squared);
  }

  // Handle the remaining elements
  int32_t partial_sum = 0;
  for (; i < dimension; i++) {
    int diff = pointer_x[i] - pointer_y[i];
    partial_sum += diff * diff;
  }

  // Reduce sum
  int32_t buffer[4];
  _mm_storeu_si128(reinterpret_cast<__m128i *>(buffer), sum);
  return static_cast<float>(buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                            partial_sum);
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