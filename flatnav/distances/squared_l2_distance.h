#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/distance_interface.h>
#include <flatnav/util/squared_l2_simd_extensions.h>
#include <functional>
#include <iostream>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension.

namespace flatnav {

struct DefaultSquaredL2 {
  static constexpr float compute(const void *x, const void *y,
                                 const size_t &dimension) {
    float *p_x = const_cast<float *>(static_cast<const float *>(x));
    float *p_y = const_cast<float *>(static_cast<const float *>(y));
    float squared_distance = 0;
    for (size_t i = 0; i < dimension; i++) {
      float difference = *p_x - *p_y;
      p_x++;
      p_y++;
      squared_distance += difference * difference;
    }
    return squared_distance;
  }
};

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

public:
  SquaredL2Distance() = default;
  explicit SquaredL2Distance(size_t dim)
      : _dimension(dim), _data_size_bytes(dim * sizeof(float)),
        _distance_computer(
            [this](const void *x, const void *y, const size_t &dimension) {
              return defaultDistanceImpl(x, y, dimension);
            }) {
    setDistanceFunction();
  }

  inline size_t getDimension() const { return _dimension; }

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

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_dimension, _data_size_bytes);

    if (Archive::is_loading::value) {
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return defaultDistanceImpl(x, y, dimension);
      };
      setDistanceFunction();
    }
  }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *destination, const void *src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nSquaredL2Distance Parameters" << std::flush;
    std::cout << "\n-----------------------------" << "\n" << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }

  void setDistanceFunction() {
#ifndef NO_SIMD_VECTORIZATION
    selectOptimalSimdStrategy();
    adjustForNonOptimalDimensions();
#endif // NO_SIMD_VECTORIZATION
  }

  void selectOptimalSimdStrategy() {
    // Start with SSE implementation
#if defined(USE_SSE)
    _distance_computer = [this](const void *x, const void *y,
                                const size_t &dimension) {
      return flatnav::util::computeL2_Sse(x, y, dimension);
    };
#endif // USE_SSE

#if defined(USE_AVX512)
    if (platformSupportsAvx512) {
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return flatnav::util::computeL2_Avx512(x, y, dimension);
      };
      return;
    }

#endif // USE_AVX512

#if defined(USE_AVX)
    if (platformSupportsAvx) {
      _distance_computer = [this](const void *x, const void *y,
                                  const size_t &dimension) {
        return flatnav::util::computeL2_Avx2(x, y, dimension);
      };
      return;
    }

#endif // USE_AVX
  }

  void adjustForNonOptimalDimensions() {
#if defined(USE_SSE)

    if (_dimension % 16 != 0) {
      if (_dimension % 4 == 0) {
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeL2_Sse4Aligned(x, y, dimension);
        };
      } else if (_dimension > 16) {
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeL2_SseWithResidual_16(x, y, dimension);
        };
      } else if (_dimension > 4) {
        _distance_computer = [this](const void *x, const void *y,
                                    const size_t &dimension) {
          return flatnav::util::computeL2_SseWithResidual_4(x, y, dimension);
        };
      }
    }
#endif // USE_SSE
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    return DefaultSquaredL2::compute(x, y, dimension);
  }
};

} // namespace flatnav
