#pragma once
#include "../DistanceInterface.h"
#include <cstddef> // for size_t

// This is the base distance function implementation for inner product distances
// on floating-point inputs.

namespace flatnav {

class InnerProductDistance : public DistanceInterface<InnerProductDistance> {
  friend class DistanceInterface<InnerProductDistance>;
  const int DISTANCE_ID = 1;

public:
  InnerProductDistance(size_t dim) {
    _dimension = dim;
    _data_size_bytes = dim * sizeof(float);
  }

  inline size_t getDimension() const { return _dimension; }

private:
  size_t _dimension;
  size_t _data_size_bytes;

  float distanceImpl(const void *x, const void *y) {
    // Default implementation of inner product distance, in case we cannot
    // support the SIMD specializations for special input _dimension sizes.
    float *p_x = (float *)x;
    float *p_y = (float *)y;
    float result = 0;
    for (size_t i = 0; i < _dimension; i++) {
      result += p_x[i] * p_y[i];
    }
    return 1.0 - result;
  }

  size_t dataSizeImpl() { return _data_size_bytes; }

  void transformDataImpl(void *dst, const void *src) {
    std::memcpy(dst, src, _data_size_bytes);
  }

  void serializeImpl(std::ofstream &out) {
    // TODO: Make this safe across machines and compilers.
    out.write(reinterpret_cast<const char *>(&DISTANCE_ID), sizeof(int));
    out.write(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
  }

  void deserializeImpl(std::ifstream &in) {
    // TODO: Make this safe across machines and compilers.
    int DISTANCE_ID_check;
    in.read(reinterpret_cast<char *>(&DISTANCE_ID_check), sizeof(int));
    if (DISTANCE_ID_check != DISTANCE_ID) {
      throw std::invalid_argument(
          "Error reading distance metric: Distance ID does not match "
          "the ID of the deserialized distance instance.");
    }
    in.read(reinterpret_cast<char *>(&_dimension), sizeof(size_t));
    _data_size_bytes = _dimension * sizeof(float);
  }
};

} // namespace flatnav