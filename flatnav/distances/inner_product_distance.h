#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/distance_interface.h>
#include <flatnav/util/inner_product_simd_extensions.h>
#include <functional>
#include <iostream>
#include <limits>

namespace flatnav {

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

struct DefaultInnerProduct {
  static constexpr float compute(const void *x, const void *y,
                                 const size_t &dimension) {
    float *p_x = static_cast<float *>(const_cast<void *>(x));
    float *p_y = static_cast<float *>(const_cast<void *>(y));
    float result = 0;
    for (size_t i = 0; i < dimension; i++) {
      result += p_x[i] * p_y[i];
    }
    return 1.0 - result;
  }
};

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {

  friend class DistanceInterface<InnerProductDistance>;
  // Enum for compile-time constant
  enum { DISTANCE_ID = 1 };

public:
  InnerProductDistance() = default;
  explicit InnerProductDistance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * sizeof(float)),
        _distance_computer(
            [this](const void *x, const void *y, const size_t &dimension) {
              return defaultDistanceImpl(x, y, dimension);
            }) {
    setDistanceFunction();
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return _distance_computer(x, y, _dimension);
  }

private:
  size_t _dimension;
  size_t _data_size_bytes;
  std::function<float(const void *, const void *, const size_t &)>
      _distance_computer;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension);

    // If loading, we need to set the data size bytes
    if (Archive::is_loading::value) {
      _data_size_bytes = _dimension * sizeof(float);
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return defaultDistanceImpl(x, y, dimension);
      };

      setDistanceFunction();
    }
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nInnerProductDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------" << "\n" << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }

  void setDistanceFunction() {
#ifndef NO_SIMD_VECTORIZATION
    selectOptimalSimdStrategy();
    adjustForNonOptimalDimensions();
#endif
  }

  void selectOptimalSimdStrategy() {
    // Start with SSE implementation
#if defined(USE_SSE)
    _distance_computer = [this](const void *x, const void *y,
                                const size_t &dimension) {
      return flatnav::util::computeIP_Sse(x, y, dimension);
    };
#endif // USE_SSE

#if defined(USE_AVX512)
    if (platformSupportsAvx512) {
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return flatnav::util::computeIP_Avx512(x, y, dimension);
      };
      return;
    }

#endif // USE_AVX512

#if defined(USE_AVX)
    if (platformSupportsAvx) {
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return flatnav::util::computeIP_Avx(x, y, dimension);
      };
      return;
    }

#endif // USE_AVX
  }

  void adjustForNonOptimalDimensions() {
#if defined(USE_SSE) || defined(USE_AVX)

    if (_dimension % 16 != 0) {
      if (_dimension % 4 == 0) {
#if defined(USE_AVX)
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeIP_Avx_4aligned(x, y, dimension);
        };
#else
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeIP_Sse_4aligned(x, y, dimension);
        };

#endif // USE_AVX
      } else if (_dimension > 16) {
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeIP_SseWithResidual_16(x, y, dimension);
        };
      } else if (_dimension > 4) {
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeIP_SseWithResidual_4(x, y, dimension);
        };
      }
    }
#endif // USE_SSE || USE_AVX
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    // Default implementation of inner product distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    return DefaultInnerProduct::compute(x, y, dimension);
  }
};

} // namespace flatnav