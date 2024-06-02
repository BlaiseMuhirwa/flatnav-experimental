#pragma once

#include <cereal/access.hpp>
#include <cereal/cereal.hpp>
#include <cstddef> // for size_t
#include <cstring> // for memcpy
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/InnerProductSimdExtensions.h>
#include <functional>
#include <iostream>
#include <limits>

namespace flatnav {

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

using util::DataType;

template <DataType data_type = DataType::float32>
struct OptimalInnerProductSimdSelector {

  static void
  adjustForNonOptimalDimensions(DistanceFunctionPtr &distance_function,
                                const size_t &dimension) {
#if defined(USE_SSE) || defined(USE_AVX)

    if (dimension % 16 != 0) {
      if (dimension % 4 == 0) {
#if defined(USE_AVX)
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeIP_Avx_4aligned, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));
#else
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeIP_Sse_4aligned, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));

#endif // USE_AVX
      } else if (dimension > 16) {
        distance_function = std::make_unique<DistanceFunction>(std::bind(
            &util::computeIP_SseWithResidual_16, std::placeholders::_1,
            std::placeholders::_2, std::cref(dimension)));

      } else if (dimension > 4) {
        distance_function = std::make_unique<DistanceFunction>(
            std::bind(&util::computeIP_SseWithResidual_4, std::placeholders::_1,
                      std::placeholders::_2, std::cref(dimension)));
      }
    }
#endif // USE_SSE || USE_AVX
  }

  static void selectInt8(DistanceFunctionPtr &distance_function,
                         const size_t &dimension) {
    (void)dimension;
    (void)distance_function;
    throw std::runtime_error("Not implemented");
  }

  static void selectUint8(DistanceFunctionPtr &distance_function,
                          const size_t &dimension) {
    (void)dimension;
    (void)distance_function;
    throw std::runtime_error("Not implemented");
  }

  static void selectFloat32(DistanceFunctionPtr &distance_function,
                            const size_t &dimension) {
#if defined(USE_SSE)
    distance_function = std::make_unique<DistanceFunction>(
        std::bind(&util::computeIP_Sse, std::placeholders::_1,
                  std::placeholders::_2, std::cref(dimension)));

#endif // USE_SSE

#if defined(USE_AVX512)
    if (platformSupportsAvx512) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeIP_Avx512, std::placeholders::_1,
                    std::placeholders::_2, std::cref(dimension)));
      adjustForNonOptimalDimensions(distance_function, dimension);
      return;
    }

#endif // USE_AVX512

#if defined(USE_AVX)
    if (platformSupportsAvx) {
      distance_function = std::make_unique<DistanceFunction>(
          std::bind(&util::computeIP_Avx, std::placeholders::_1,
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
    case DataType::uint8:
      selectUint8(distance_function, dimension);
      break;
    case DataType::int8:
      selectInt8(distance_function, dimension);
      break;
    default:
      throw std::runtime_error("Unsupported data type");
    }
  }
};

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {

  friend class DistanceInterface<InnerProductDistance>;
  // Enum for compile-time constant
  enum { DISTANCE_ID = 1 };

public:
  InnerProductDistance() = default;
  InnerProductDistance(size_t dim, DataType data_type = DataType::float32)
      : _data_type(data_type), _dimension(dim),
        _data_size_bytes(dim * flatnav::util::size(data_type)),
        _distance_computer(nullptr) {}

  template <DataType data_type = DataType::float32>
  static std::unique_ptr<InnerProductDistance> create(size_t dim) {
    if (data_type == DataType::undefined) {
      throw std::runtime_error("Undefined data type");
    }

    return std::make_unique<InnerProductDistance>(dim, data_type);
  }

  float distanceImpl(const void *x, const void *y,
                     bool asymmetric = false) const {
    (void)asymmetric;
    return (*_distance_computer)(x, y, _dimension);
  }

  inline constexpr DataType dataTypeImpl() const { return _data_type; }

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
        &InnerProductDistance::defaultDistanceImpl, this, std::placeholders::_1,
        std::placeholders::_2, std::placeholders::_3));

    OptimalInnerProductSimdSelector<data_type>::select(_distance_computer,
                                                       _dimension);
  }

private:
  DataType _data_type;
  size_t _dimension;
  size_t _data_size_bytes;
  DistanceFunctionPtr _distance_computer;

  friend class cereal::access;

  template <typename Archive> void serialize(Archive &ar) {
    ar(_data_type, _dimension, _data_size_bytes);

    // If loading, we need to set the data size bytes
    if (Archive::is_loading::value) {
      setDistanceFunctionWithType();
    }
  }

  inline size_t getDimension() const { return _dimension; }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void getSummaryImpl() {
    std::cout << "\nInnerProductDistance Parameters" << std::flush;
    std::cout << "\n-----------------------------"
              << "\n"
              << std::flush;
    std::cout << "Dimension: " << _dimension << "\n" << std::flush;
  }

  float defaultDistanceImpl(const void *x, const void *y,
                            const size_t &dimension) const {
    // Default implementation of inner product distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = static_cast<float *>(const_cast<void *>(x));
    float *p_y = static_cast<float *>(const_cast<void *>(y));

    float result = 0;
    for (size_t i = 0; i < dimension; i++) {
      result += p_x[i] * p_y[i];
    }
    return 1.0f - result;
  }
};

} // namespace flatnav