

#include "../BaseProductQuantization.h"
#include <flatnav/DistanceInterface.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>

using flatnav::quantization::ProductQuantizer;

namespace flatnav::quantization {

const std::vector<float> generateRandomVectors(uint32_t count, uint32_t dim) {
  std::vector<float> vectors(dim * count);

  std::mt19937 gen(1234);
  std::uniform_real_distribution<float> dis(-1.0, 1.0);

  for (uint32_t i = 0; i < dim * count; i++) {
    vectors[i] = dis(gen);
  }
  return vectors;
}

TEST(ProductQuantizationTest, TestComputeCodes) {
  // Dimensionality of the input vectors
  const uint32_t dim = 64;
  // Number of subquantizers
  const uint32_t M = 4;
  // Number of bits per subvector index
  const uint32_t nbits = 8;
  // Number of testing vectors
  const uint32_t num_vectors = 1000;

  auto distance = std::make_unique<SquaredL2Distance>(dim);
  ProductQuantizer<SquaredL2Distance> pq(/* dist = */ std::move(distance),
                                         /* dim = */ dim, /* M = */ M,
                                         /* nbits = */ nbits);

  // Generate random vectors
  auto testing_vectors =
      generateRandomVectors(/* count = */ num_vectors, /* dim = */ dim);

  // Train the quantizer
  pq.train(/* vectors = */ testing_vectors.data(),
           /* num_vectors = */ testing_vectors.size());

  auto code_size = pq.getCodeSize();
  std::cout << "[INFO] Code size: " << code_size << std::endl;

  // Encode
  std::vector<uint8_t> codes(code_size * num_vectors);
  pq.computePQCodes(/* vectors = */ testing_vectors.data(),
                    /* codes = */ codes.data(), /* n = */ num_vectors);

  // Decode
  std::vector<float> decoded_vectors(dim * num_vectors);
  pq.decode(/* code = */ codes.data(), /* vectors = */ decoded_vectors.data(),
            /* n = */ num_vectors);

  // Encode the second time
  std::vector<uint8_t> second_encoding(code_size * num_vectors);

  pq.computePQCodes(/* vectors = */ testing_vectors.data(),
                    /* codes = */ second_encoding.data(),
                    /* n = */ num_vectors);

  // Check that the second encoding is the same as the first one
  for (uint32_t i = 0; i < codes.size(); i++) {
    ASSERT_EQ(codes[i], second_encoding[i]);
  }
}

} // namespace flatnav::quantization