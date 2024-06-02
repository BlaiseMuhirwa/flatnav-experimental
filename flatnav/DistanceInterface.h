#pragma once

#include <cereal/access.hpp>
#include <cstddef> // for size_t
#include <flatnav/util/Datatype.h>
#include <fstream> // for ifstream, ofstream
#include <functional>
#include <iostream>

namespace flatnav {

using util::DataType;
typedef std::function<float(const void *, const void *, const size_t &)>
    DistanceFunction;

typedef std::unique_ptr<DistanceFunction> DistanceFunctionPtr;

enum class METRIC_TYPE { EUCLIDEAN, INNER_PRODUCT };

// We use the CRTP to implement static polymorphism on the distance. This is
// done to allow for metrics and distance functions that support arbitrary
// pre-processing (such as quantization etc) without having to call the
// distance function through a pointer or virtual function call.
// CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern

template <typename T> class DistanceInterface {
public:
  // The asymmetric flag is used to indicate whether the distance function
  // is between two database vectors (symmetric) or between a database vector
  // and a query vector. For regular distances (l2, inner product), there is
  // no difference between the two. However, for quantization techniques, such
  // as product quantization, the two distance modes are different.
  float distance(const void *x, const void *y, bool asymmetric = false) {
    return static_cast<T *>(this)->distanceImpl(x, y, asymmetric);
  }

  // Returns the dimension of the input data.
  size_t dimension() { return static_cast<T *>(this)->getDimension(); }

  // Returns the size, in bytes, of the transformed data representation.
  size_t dataSize() { return static_cast<T *>(this)->dataSizeImpl(); }

  // Prints the parameters of the distance function.
  void getSummary() { static_cast<T *>(this)->getSummaryImpl(); }

  inline constexpr DataType dataType() {
    return static_cast<T *>(this)->dataTypeImpl();
  }

  // This transforms the data located at src into a form that is writeable
  // to disk / storable in RAM. For distance functions that don't
  // compress the input, this just passses through a copy from src to
  // destination. However, there are functions (e.g. with quantization) where
  // the in-memory representation is not the same as the raw input.
  void transformData(void *destination, const void *src) {
    static_cast<T *>(this)->transformDataImpl(destination, src);
  }

  // Serializes the distance function to disk.
  template <typename Archive> void serialize(Archive &archive) {
    static_cast<T *>(this)->template serialize<Archive>(archive);
  }
};

} // namespace flatnav