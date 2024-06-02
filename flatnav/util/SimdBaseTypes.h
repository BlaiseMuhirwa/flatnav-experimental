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

#include <flatnav/util/Macros.h>

namespace flatnav::util {

template <size_t N> struct MaskRepr {};
template <> struct MaskRepr<2> { using type = uint8_t; };
template <> struct MaskRepr<4> { using type = uint8_t; };
template <> struct MaskRepr<8> { using type = uint8_t; };
template <> struct MaskRepr<16> { using type = uint16_t; };
template <> struct MaskRepr<32> { using type = uint32_t; };
template <> struct MaskRepr<64> { using type = uint64_t; };

template <typename T> struct MaskIntrinsic {};
template <> struct MaskIntrinsic<uint8_t> { using mask_type = __mmask8; };
template <> struct MaskIntrinsic<uint16_t> { using mask_type = __mmask16; };
template <> struct MaskIntrinsic<uint32_t> { using mask_type = __mmask32; };
template <> struct MaskIntrinsic<uint64_t> { using mask_type = __mmask64; };

// Given a length `N`, obtain an appropriate integer type used as a mask for `N`
// lanes in an AVX vector operation.
template <size_t N> using mask_repr_t = typename MaskRepr<N>::type;

// Given an unsigned integer type, return the corresponding mask type
template <typename T>
using mask_intrinsic_t = typename MaskIntrinsic<T>::mask_type;

// Given a length `N`, obtain an appropriate mask intrinsic type.
template <size_t N>
using mask_intrinsic_from_length = mask_intrinsic_t<mask_repr_t<N>>;

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

  void storeu(void *pointer) const {
    _mm_storeu_si128((__m128i *)pointer, _int);
  }
  void store(void *pointer) const {
    // This requires 16-byte alignment for pointer
    _mm_store_si128((__m128i *)pointer, _int);
  }

  void clear() { _int = _mm_setzero_si128(); }

  void loadu(const void *pointer) {
    _float = _mm_loadu_ps((const float *)pointer);
  }

  float reduce_add() const {
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

  bool operator=(const simd128bit &other) {
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

  simd4float32 operator+(const simd4float32 &other) const {
    __m128 result = _mm_add_ps(_float, other._float);
    return simd4float32(result);
  }

  simd4float32 operator*(const simd4float32 &other) const {
    __m128 result = _mm_mul_ps(_float, other._float);
    return simd4float32(result);
  }

  simd4float32 operator/(const simd4float32 &other) const {
    __m128 result = _mm_div_ps(_float, other._float);
    return simd4float32(result);
  }

  simd4float32 operator-(const simd4float32 &other) const {
    __m128 result = _mm_sub_ps(_float, other._float);
    return simd4float32(result);
  }

  simd4float32 &operator+=(const simd4float32 &other) {
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

  void loadu(const void *pointer) {
    _float = _mm256_loadu_ps((const float *)pointer);
  }

  // This is supposed to be faster than using _mm256_store_ps to store
  // a simd256bit type into a float array of size 8 and then performing
  // scalar addition.
  // TODO: Actually run some benchmarks to verify this.
  float reduce_add() const {
    // AVX doesn't have an equivalent intrinsic for _mm512_reduce_add_ps, but
    // we can achieve the same result as follows:
    __m256 sum = _mm256_hadd_ps(_float, _float);
    sum = _mm256_hadd_ps(sum, sum);

    // Move the sum to the lower half of the register
    __m128i low128 = _mm256_extractf128_si256(_mm256_castps_si256(sum), 0);

    // Perform a final horizontal add if necessary (now on 128-bit lane)
    // The elements in the lower 128 bits are now in the first and second
    // positions
    __m128 hsum128 =
        _mm_hadd_ps(_mm_castsi128_ps(low128), _mm_castsi128_ps(low128));

    // Extract the first element from the 128-bit lane
    return _mm_cvtss_f32(hsum128);
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

  simd8float32 operator+(const simd8float32 &other) const {
    __m256 result = _mm256_add_ps(_float, other._float);
    return simd8float32(result);
  }

  simd8float32 operator-(const simd8float32 &other) const {
    __m256 result = _mm256_sub_ps(_float, other._float);
    return simd8float32(result);
  }

  simd8float32 operator*(const simd8float32 &other) const {
    __m256 result = _mm256_mul_ps(_float, other._float);
    return simd8float32(result);
  }

  simd8float32 operator/(const simd8float32 &other) const {
    __m256 result = _mm256_div_ps(_float, other._float);
    return simd8float32(result);
  }

  simd8float32 &operator+=(const simd8float32 &other) {
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
  // void store(void *pointer) const {
  //   // This requires 64-byte alignment for pointer
  //   _mm512_store_si512((__m512i *)pointer, _int);
  // }

  void store(void *pointer) const {
    // This requires 64-byte alignment for pointer
    _mm512_store_ps((float *)pointer, _float);
  }

  void clear() { _int = _mm512_setzero_si512(); }

  void loadu(const void *pointer) {
    _float = _mm512_loadu_ps((const float *)pointer);
  }

  float reduce_add() const { return _mm512_reduce_add_ps(_float); }
};

struct simd64int8 : public simd512bit {

  // Default constructor to zero-initialize the vector
  simd64int8() : simd512bit(_mm512_setzero_si512()) {}

  // Construct from __m512i
  explicit simd64int8(__m512i x) : simd512bit(x) {}
  explicit simd64int8(const void *x) : simd512bit(x) {}

  // Load from memory location (unaligned)
  explicit simd64int8(const int8_t *x)
      : simd512bit(_mm512_loadu_si512((__m512i *)x)) {}

  // Set all elements to a specific value
  explicit simd64int8(int8_t x) : simd512bit(_mm512_set1_epi8(x)) {}

  inline constexpr __m512i get() const { return _int; }

  // Basic arithmetic operations using AVX512BW intrinsics
  simd64int8 operator+(const simd64int8 &other) const {
    __m512i result = _mm512_add_epi8(_int, other._int);
    return simd64int8(result);
  }

  simd64int8 operator-(const simd64int8 &other) const {
    __m512i result = _mm512_sub_epi8(_int, other._int);
    return simd64int8(result);
  }

  // in-place addition
  simd64int8 &operator+=(const simd64int8 &other) {
    _int = _mm512_add_epi8(_int, other._int);
    return *this;
  }

  // Multiply by another vector, producing 32-bit integers, and store the low 16
  // bits of the intermediate result

  // storing and loading
  void storeu(int8_t *pointer) const {
    _mm512_storeu_si512((__m512i *)pointer, _int);
  }

  void loadu(const int8_t *pointer) {
    _int = _mm512_loadu_si512((__m512i *)pointer);
  }
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

  simd16float32 operator+(const simd16float32 &other) const {
    __m512 result = _mm512_add_ps(_float, other._float);
    return simd16float32(result);
  }

  simd16float32 operator*(const simd16float32 &other) const {
    __m512 result = _mm512_mul_ps(_float, other._float);
    return simd16float32(result);
  }

  simd16float32 operator/(const simd16float32 &other) const {
    __m512 result = _mm512_div_ps(_float, other._float);
    return simd16float32(result);
  }

  simd16float32 &operator+=(const simd16float32 &other) {
    _float = _mm512_add_ps(_float, other._float);
    return *this;
  }

  simd16float32 operator-(const simd16float32 &other) const {
    __m512 result = _mm512_sub_ps(_float, other._float);
    return simd16float32(result);
  }
};

#endif // USE_AVX512

} // namespace flatnav::util
