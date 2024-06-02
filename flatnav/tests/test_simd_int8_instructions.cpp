#include "gtest/gtest.h"
#include <cassert>
#include <cstdio> // for remove
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/Datatype.h>
#include <gtest/gtest.h>
#include <random>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::SquaredL2Distance;
using flatnav::util::DataType;

namespace flatnav::testing {

static const uint32_t INDEXED_VECTORS = 2;
static const uint32_t VEC_DIM = 100;

void printVector(void *vector, uint32_t dim) {
  for (uint32_t i = 0; i < dim; i++) {
    printf("%d ", ((int8_t *)vector)[i]);
  }
  printf("\n");
}

std::vector<int8_t> generateTestVectors(uint32_t num_vectors, uint32_t dim) {
  std::vector<int8_t> vectors(num_vectors * dim);
  for (uint32_t i = 0; i < num_vectors * dim; i++) {
    vectors[i] = (int8_t)(rand() % 256);
  }
  return vectors;
}

TEST(SIMD_INT8_TESTS, TestSquaredL2Distance) {
  // This test checks that the computed distance with the int8_t simd
  // instructions is the same as the float simd instructions.
  auto vectors = generateTestVectors(INDEXED_VECTORS, VEC_DIM);

  // Make a copy that casts each value to float.
  std::vector<float> float_vectors(vectors.begin(), vectors.end());

  auto int8_distance = SquaredL2Distance::create<DataType::int8>(VEC_DIM);
  int8_distance->setDistanceFunction<DataType::int8>();

  auto float_distance = SquaredL2Distance::create(VEC_DIM);
  float_distance->setDistanceFunction();

  for (uint32_t i = 0; i < INDEXED_VECTORS - 1; i++) {
    for (uint32_t j = i + 1; j < INDEXED_VECTORS; j++) {
      int8_t *first_int8_vector = vectors.data() + (VEC_DIM * i);
      int8_t *second_int8_vector = vectors.data() + (VEC_DIM * j);

      printVector(first_int8_vector, VEC_DIM);
      printVector(second_int8_vector, VEC_DIM);

      float *first_float_vector = float_vectors.data() + (VEC_DIM * i);
      float *second_float_vector = float_vectors.data() + (VEC_DIM * j);

      ASSERT_FLOAT_EQ(
          int8_distance->distanceImpl(first_int8_vector, second_int8_vector),
          float_distance->distanceImpl(first_float_vector,
                                       second_float_vector));
    }
  }
}

} // namespace flatnav::testing
