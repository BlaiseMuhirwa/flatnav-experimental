
#include "../CentroidsGenerator.h"
#include "gtest/gtest.h"
#include <cstdint>

using flatnav::quantization::CentroidsGenerator;

namespace flatnav::tests {

TEST(CentroidsGeneratorTest, TestCentroidsInitialization) {
  // Initialize parameters
  uint64_t dim = 2;
  uint64_t num_centroids = 2;
  uint64_t n = 5;

  // Create some mock data
  std::vector<float> vectors = {1.0, 2.0, 3.0, 4.0, 5.0,
                                6.0, 7.0, 8.0, 9.0, 10.0};

  CentroidsGenerator generator(/* dim = */ dim,
                               /* num_centroids= */ num_centroids);

  generator.generateCentroids(/* vectors = */ vectors.data(),
                              /* vec_weights = */ NULL,
                              /* n = */ n);

  // Check if the correct number of centroids have been generated
  ASSERT_EQ(generator.centroids().size(), num_centroids * dim);
}

TEST(CentroidsGeneratorTest, TestCentroidsValues) {
  // Initialize parameters
  uint64_t dim = 2;
  uint64_t num_centroids = 2;
  uint64_t n = 4;

  // Create some mock data
  std::vector<float> vectors = {1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 7.0, 7.0};

  // Create an instance of CentroidsGenerator
  CentroidsGenerator generator(/* dim = */ dim,
                               /* num_centroids = */ num_centroids,
                               /* num_iterations = */ 1000);

  generator.generateCentroids(/* vectors= */ vectors.data(),
                              /* vec_weights = */ NULL, /* n = */ n);

  std::vector<float> centroids = generator.centroids();
  ASSERT_EQ(centroids.size(), num_centroids * dim);

  // TODO: Check if the correct centroids have been computed as well.
}

} // namespace flatnav::tests