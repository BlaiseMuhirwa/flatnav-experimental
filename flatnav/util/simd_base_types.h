/**
 * Adopted from
 * https://github.com/facebookresearch/faiss/blob/main/faiss/utils/simdlib_avx2.h
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flatnav/util/macros.h>

namespace flatnav::util {

#if defined(USE_SSE)
struct simd128bit {
  union {
    __m128 _float;
    __m128i _int;
  };

  explicit simd128bit(__m128i x) : _int(x) {}
  explicit simd128bit(__m128 x) : _float(x) {}
  explicit simd128bit(const void *x)
      : _int(_mm_loadu_si128((const __m128i *)x)) {}

  inline void storeu(void *pointer) const {
    _mm_storeu_si128((__m128i *)pointer, _int);
  }
  inline void store(void *pointer) const {
    // This requires 16-byte alignment for pointer
    _mm_store_si128((__m128i *)pointer, _int);
  }

  void clear() { _int = _mm_setzero_si128(); }

  inline void loadu(const void *pointer) {
    _float = _mm_loadu_ps((const float *)pointer);
  }

  inline float reduce_add() const {
#if defined(USE_SSE3)
    // _mm_hadd_ps is only available in SSE3
    __m128 sum = _mm_hadd_ps(_float, _float);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
#else
    // We will store the result in a float array and then sum the elements
    // This is supposed to be less efficient that using the _mm_hadd_ps
    // intrinsic.
    float result[4];
    _mm_storeu_ps(result, _float);
    return result[0] + result[1] + result[2] + result[3];
#endif
  }

  bool operator==(const simd128bit &other) {
    const __m128i eq = _mm_cmpeq_epi32(_int, other._int);
    return _mm_movemask_epi8(eq) == 0xffffU;
  }
};

struct simd4float32 : public simd128bit {
  simd4float32() : simd128bit(_mm_setzero_ps()) {}
  explicit simd4float32(__m128 x) : simd128bit(x) {}
  explicit simd4float32(const void *x) : simd128bit(x) {}

  // Set all elements to x
  explicit simd4float32(float x) : simd128bit(_mm_set1_ps(x)) {}

  explicit simd4float32(const float *x) : simd128bit(_mm_loadu_ps(x)) {}

  explicit simd4float32(float x0, float x1, float x2, float x3)
      : simd128bit(_mm_setr_ps(x3, x2, x1, x0)) {}

  inline simd4float32 operator+(const simd4float32 &other) const {
    __m128 result = _mm_add_ps(_float, other._float);
    return simd4float32(result);
  }

  inline simd4float32 operator*(const simd4float32 &other) const {
    __m128 result = _mm_mul_ps(_float, other._float);
    return simd4float32(result);
  }

  inline simd4float32 operator/(const simd4float32 &other) const {
    __m128 result = _mm_div_ps(_float, other._float);
    return simd4float32(result);
  }

  inline simd4float32 operator-(const simd4float32 &other) const {
    __m128 result = _mm_sub_ps(_float, other._float);
    return simd4float32(result);
  }

  inline simd4float32 &operator+=(const simd4float32 &other) {
    _float = _mm_add_ps(_float, other._float);
    return *this;
  }

  bool operator==(const simd4float32 &other) const {
    const __m128i eq =
        _mm_castps_si128(_mm_cmp_ps(_float, other._float, _CMP_EQ_OQ));
    return _mm_movemask_epi8(eq) == 0xffffU;
  }
};

#endif // USE_SSE

#if defined(USE_AVX)
struct simd256bit {
  union {
    __m256 _float;
    __m256i _int;
  };

  explicit simd256bit(__m256i x) : _int(x) {}
  explicit simd256bit(__m256 x) : _float(x) {}
  explicit simd256bit(const void *x)
      : _int(_mm256_loadu_si256((const __m256i *)x)) {}

  void storeu(void *pointer) const {
    _mm256_storeu_si256((__m256i *)pointer, _int);
  }
  void store(void *pointer) const {
    // This requires 32-byte alignment for pointer
    _mm256_store_si256((__m256i *)pointer, _int);
  }

  void clear() { _int = _mm256_setzero_si256(); }

  __m128 get_low() const { return _mm256_extractf128_ps(_float, 0); }
  __m128 get_high() const { return _mm256_extractf128_ps(_float, 1); }

  inline void loadu(const void *pointer) {
    _float = _mm256_loadu_ps((const float *)pointer);
  }

  // clang-format off
  /**
   * Computes the sum of all numbers in a SIMD register using AVX instructions.
   * This is actually a bit faster than doing the following
   * ```cpp
   * float result[8];
   * _mm256_storeu_ps(result, _float);
   * return result[0] + result[1] + result[2] + result[3] + result[4] +
   * result[5] + result[6] + result[7];
   * ```
   * AVX512 has a dedicated instruction for this, but AVX/AVX2 do not.
   * Example:
   * If _float = [a, b, c, d, e, f, g, h], the following is what each
   * instruction returns:
   *  _mm256_hadd_ps(_float, _float) = [a+b, c+d, a+b, c+d, e+f, g+h, e+f, g+h] 
   * _mm256_hadd_ps(sum, sum) = [a+b+c+d, a+b+c+d, e+f+g+h, e+f+g+h, ...] 
   * _mm256_extractf128_ps(sum, 0) = [a+b+c+d, a+b+c+d, ...]
   * _mm256_extractf128_ps(sum, 1) = [e+f+g+h, e+f+g+h, ...]
   * _mm_add_ps(low128, high128) = [(a+b+c+d)+(e+f+g+h), (a+b+c+d)+(e+f+g+h), ...] 
   * _mm_cvtss_f32(final_sum) = a+b+c+d+e+f+g+h
   *
   */
  float reduce_add() const {
    // Horizontal add within each 128-bit lane
    __m256 sum = _mm256_hadd_ps(_float, _float);
    sum = _mm256_hadd_ps(sum, sum);

    // Extract both 128-bit parts and add them together
    __m128 low128 = _mm256_extractf128_ps(sum, 0);
    __m128 high128 = _mm256_extractf128_ps(sum, 1);
    // Adds corresponding elements of low128 and high128
    __m128 final_sum = _mm_add_ps(low128, high128);

    // Extract the first element, which now holds the total sum of all eight
    // original elements
    return _mm_cvtss_f32(final_sum);
  }
};

// Vector of 8 32-bit floats
struct simd8float32 : public simd256bit {
  simd8float32() : simd256bit(_mm256_setzero_ps()) {}
  explicit simd8float32(__m256 x) : simd256bit(x) {}
  explicit simd8float32(const void *x) : simd256bit(x) {}
  explicit simd8float32(float x) : simd256bit(_mm256_set1_ps(x)) {}

  explicit simd8float32(const float *x) : simd256bit(_mm256_loadu_ps(x)) {}

  explicit simd8float32(float x0, float x1, float x2, float x3, float x4,
                        float x5, float x6, float x7)
      : simd256bit(_mm256_setr_ps(x7, x6, x5, x4, x3, x2, x1, x0)) {}

  inline simd8float32 operator+(const simd8float32 &other) const {
    __m256 result = _mm256_add_ps(_float, other._float);
    return simd8float32(result);
  }

  inline simd8float32 operator-(const simd8float32 &other) const {
    __m256 result = _mm256_sub_ps(_float, other._float);
    return simd8float32(result);
  }

  inline simd8float32 operator*(const simd8float32 &other) const {
    __m256 result = _mm256_mul_ps(_float, other._float);
    return simd8float32(result);
  }

  inline simd8float32 operator/(const simd8float32 &other) const {
    __m256 result = _mm256_div_ps(_float, other._float);
    return simd8float32(result);
  }

  inline simd8float32 &operator+=(const simd8float32 &other) {
    _float = _mm256_add_ps(_float, other._float);
    return *this;
  }
};

#endif // USE_AVX

#if defined(USE_AVX512)
struct simd512bit {
  union {
    __m512 _float;
    __m512i _int;
  };

  explicit simd512bit(__m512i x) : _int(x) {}
  explicit simd512bit(__m512 x) : _float(x) {}
  explicit simd512bit(const void *x)
      : _int(_mm512_loadu_si512((const __m512i *)x)) {}

  // constructs from lower and upper halves
  // NOTE: This implementation might be wrong!
  simd512bit(const simd256bit &lo, const simd256bit &hi) {}

  void storeu(void *pointer) const {
    _mm512_storeu_si512((__m512i *)pointer, _int);
  }

  inline void store(void *pointer) const {
    // This requires 64-byte alignment for pointer
    _mm512_store_ps((float *)pointer, _float);
  }

  inline void clear() { _int = _mm512_setzero_si512(); }

  inline void loadu(const void *pointer) {
    _float = _mm512_loadu_ps((const float *)pointer);
  }

  inline float reduce_add() const { return _mm512_reduce_add_ps(_float); }
};

struct simd16float32 : public simd512bit {
  simd16float32() : simd512bit(_mm512_setzero_ps()) {}
  explicit simd16float32(__m512 x) : simd512bit(x) {}
  explicit simd16float32(const void *x) : simd512bit(x) {}

  // Set all elements to x
  explicit simd16float32(float x) : simd512bit(_mm512_set1_ps(x)) {}

  explicit simd16float32(const float *x) : simd512bit(_mm512_loadu_ps(x)) {}

  explicit simd16float32(float x0, float x1, float x2, float x3, float x4,
                         float x5, float x6, float x7, float x8, float x9,
                         float x10, float x11, float x12, float x13, float x14,
                         float x15)
      : simd512bit(_mm512_setr_ps(x15, x14, x13, x12, x11, x10, x9, x8, x7, x6,
                                  x5, x4, x3, x2, x1, x0)) {}

  // Constructs from lower and upper halves
  simd16float32(const simd8float32 &lo, const simd8float32 &hi)
      : simd512bit(lo, hi) {}

  inline simd16float32 operator+(const simd16float32 &other) const {
    __m512 result = _mm512_add_ps(_float, other._float);
    return simd16float32(result);
  }

  inline simd16float32 operator*(const simd16float32 &other) const {
    __m512 result = _mm512_mul_ps(_float, other._float);
    return simd16float32(result);
  }

  inline simd16float32 operator/(const simd16float32 &other) const {
    __m512 result = _mm512_div_ps(_float, other._float);
    return simd16float32(result);
  }

  inline simd16float32 &operator+=(const simd16float32 &other) {
    _float = _mm512_add_ps(_float, other._float);
    return *this;
  }

  inline simd16float32 operator-(const simd16float32 &other) const {
    __m512 result = _mm512_sub_ps(_float, other._float);
    return simd16float32(result);
  }
};

#endif // USE_AVX512

} // namespace flatnav::util
