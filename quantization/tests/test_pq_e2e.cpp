

#include <flatnav/distances/squared_l2_distance.h>
#include <gtest/gtest.h>
#include <memory>
#include <quantization/product_quantization.h>
#include <random>

using flatnav::quantization::ProductQuantizer;

namespace flatnav::quantization {

std::vector<float> generateRandomVectors(uint32_t count, uint32_t dim) {
  std::vector<float> vectors(dim * count, 0.F);

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

  ProductQuantizer<SquaredL2Distance> pq(/* dim = */ dim, /* M = */ M,
                                         /* nbits = */ nbits);

  // Generate random vectors
  std::vector<float> testing_vectors =
      generateRandomVectors(/* count = */ num_vectors, /* dim = */ dim);

  // Train the quantizer
  pq.train(/* vectors = */ testing_vectors.data(),
           /* num_vectors = */ num_vectors);

  auto code_size = pq.getCodeSize();
  std::cout << "[INFO] Code size: " << code_size << std::endl;

  // Encode
  uint8_t *codes = new uint8_t[code_size * num_vectors];

  pq.computePQCodes(/* vectors = */ testing_vectors.data(),
                    /* codes = */ codes, /* n = */ num_vectors);

  // Decode
  std::vector<float> decoded_vectors(dim * num_vectors);
  pq.decode(/* code = */ codes, /* vectors = */ decoded_vectors.data(),
            /* n = */ num_vectors);

  // Encode the second time
  uint8_t *second_encoding = new uint8_t[code_size * num_vectors];

  pq.computePQCodes(/* vectors = */ testing_vectors.data(),
                    /* codes = */ second_encoding,
                    /* n = */ num_vectors);

  // Check that the second encoding is the same as the first one
  for (uint32_t i = 0; i < code_size * num_vectors; i++) {
    ASSERT_EQ(codes[i], second_encoding[i]);
  }

  delete[] codes;
  delete[] second_encoding;
}

} // namespace flatnav::quantization