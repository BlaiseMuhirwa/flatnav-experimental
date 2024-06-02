#pragma once
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/SquaredL2SimdExtensions.h>
#include <functional>
#include <iostream>
#include <type_traits>

// This is the base distance function implementation for the L2 distance on
// floating-point inputs. We provide specializations that use SIMD when
// supported by the compiler and compatible with the input _dimension.

namespace flatnav {

using util::DataType;

template <DataType data_type = DataType::float32> struct OptimalL2SimdSelector {

  static void
  adjustForNonOptimalDimensions(DistanceFunctionPtr &distance_function,
                                const size_t &dimension) {
#if defined(USE_SSE)
    if (dimension % 16 != 0) {
      if (dimension % 4 == 0) {
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeL2_Sse4Aligned, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));

      } else if (dimension > 16) {
        distance_function = std::make_unique<DistanceFunction>(std::bind(
            &util::computeL2_SseWithResidual_16, std::placeholders::_1,
            std::placeholders::_2, std::cref(dimension)));

      } else if (dimension > 4) {
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeL2_SseWithResidual_4, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));
      }
    }
#endif // USE_SSE
  }

  static void selectInt8(DistanceFunctionPtr &distance_function,
                         const size_t &dimension) {
#if defined(USE_AVX512)
    if (platformSupportsAvx512) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Avx512_int8, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));
    }
#endif // USE_AVX512
  }

  static void selectUint8(DistanceFunctionPtr &distance_function,
                          const size_t &dimension) {
    (void)distance_function;
    (void)dimension;
    throw std::runtime_error("Not yet implemented!");
  }

  static void selectFloat32(DistanceFunctionPtr &distance_function,
                            const size_t &dimension) {
#if defined(USE_SSE)
    distance_function = std::make_unique<DistanceFunction>(
        std::bind(&util::computeL2_Sse, std::placeholders::_1,
                  std::placeholders::_2, std::cref(dimension)));
#endif // USE_SSE

#if defined(USE_AVX512)
    if (platformSupportsAvx512) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Avx512, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));

      adjustForNonOptimalDimensions(distance_function, dimension);
      return;
    }
#endif // USE_AVX512

#if defined(USE_AVX)
    if (platformSupportsAvx) {

      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeL2_Avx2, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));
      adjustForNonOptimalDimensions(distance_function, dimension);
      return;
    }
#endif // USE_AVX

    adjustForNonOptimalDimensions(distance_function, dimension);
  }

  /**
   * @brief Select the optimal distance function based on the input dimension
   * @param dimension The dimension of the input data
   *
   * @note There are different SIMD functions for float32, int8_t and uint8_t.
   * This is why we are templating this class on the data type.
   */
  static void select(DistanceFunctionPtr &distance_function,
                     const size_t &dimension) {
    switch (data_type) {
    case DataType::float32:
      selectFloat32(distance_function, dimension);
      break;
    case DataType::int8:
      selectInt8(distance_function, dimension);
      break;
    case DataType::uint8:
      selectUint8(distance_function, dimension);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
  }
};

class SquaredL2Distance : public DistanceInterface<SquaredL2Distance> {

  friend class DistanceInterface<SquaredL2Distance>;
  enum { DISTANCE_ID = 0 };

public:
  SquaredL2Distance() = default;
  SquaredL2Distance(size_t dim, DataType data_type = DataType::float32)
      : _data_type(data_type), _dimension(dim),
        _data_size_bytes(dim * util::size(data_type)) {}
  // _distance_computer(nullptr) {}

  template <DataType data_type = DataType::float32>
  static std::unique_ptr<SquaredL2Distance> create(size_t dim) {
    return std::make_unique<SquaredL2Distance>(dim, data_type);
  }

  inline constexpr size_t getDimension() const { return _dimension; }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return (*_distance_computer)(x, y, _dimension);
  }

  /**
   * @brief Dispatcher for templating setDistanceFunction the right data type
   */
  void setDistanceFunctionWithType() {
#ifndef NO_SIMD_VECTORIZATION
    switch (_data_type) {
    case DataType::float32:
      setDistanceFunction<DataType::float32>();
      break;
    case DataType::uint8:
      setDistanceFunction<DataType::uint8>();
      break;

    case DataType::int8:
      setDistanceFunction<DataType::int8>();
      break;

    default:
      throw std::runtime_error("Unsupported data type");
    }
#endif // NO_SIMD_VECTORIZATION
  }

  template <DataType data_type = DataType::float32> void setDistanceFunction() {
    _distance_computer = std::make_unique<DistanceFunction>(std::bind(
        &SquaredL2Distance::defaultDistanceImpl, this, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3));

    OptimalL2SimdSelector<data_type>::select(_distance_computer, _dimension);
  }

  inline constexpr DataType dataTypeImpl() const { return _data_type; }

private:
  DataType _data_type;
  size_t _dimension;
  size_t _data_size_bytes;
  DistanceFunctionPtr _distance_computer;

  friend class ::cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_data_type, _dimension, _data_size_bytes);

    if (Archive::is_loading::value) {
      setDistanceFunctionWithType();
    }
  }

  inline size_t dataSizeImpl() { return _data_size_bytes; }

  inline void transformDataImpl(void *destination, const void *src) {
    std::memcpy(destination, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nSquaredL2Distance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
    std::cout << "Data Type: " << util::name(_data_type) << "\n" << std::flush;
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    // Default implementation of squared-L2 distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.

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

} // namespace flatnav
