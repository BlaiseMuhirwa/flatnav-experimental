
#include "../BaseProductQuantization.h"
#include <cstdlib>
#include <gtest/gtest.h>
#include <vector>

using flatnav::quantization::PQCodeManager;
;

namespace flatnav::tests {

const std::vector<uint64_t> generateRandomVector(uint32_t random_vec_size) {
  std::vector<uint64_t> random_vec;
  for (uint32_t i = 0; i < random_vec_size; ++i) {
    random_vec.push_back(rand());
  }
  return random_vec;
}

TEST(PQCodeManagerTest, TestEncodingAndDecodingUint8) {
  const int nsubcodes = 100;
  const uint64_t mask = 0xFF;

  auto testing_vector = generateRandomVector(/* vec_size = */ nsubcodes);

  std::unique_ptr<uint8_t[]> codes(new uint8_t[nsubcodes]);

  PQCodeManager<uint8_t> code_manager(/* code = */ codes.get(),
                                      /* nbits = */ 8);

  for (const auto &val : testing_vector) {
    code_manager.encode(val & mask);
  }
  code_manager.jumpToStart();

  for (int i = 0; i < nsubcodes; ++i) {
    uint64_t decoded = code_manager.decode();
    ASSERT_EQ(decoded, testing_vector[i] & mask);
  }
}

TEST(PQCodeManagerTest, TestEncodingAndDecodingUint16) {
  const int nsubcodes = 100;
  const uint64_t mask = 0xFFFF;

  auto testing_vector = generateRandomVector(/* vec_size = */ nsubcodes);

  std::unique_ptr<uint8_t[]> codes(new uint8_t[2 * nsubcodes]);

  PQCodeManager<uint16_t> code_manager(/* code = */ codes.get(),
                                       /* nbits = */ 16);

  for (const auto &val : testing_vector) {
    code_manager.encode(val & mask);
  }
  code_manager.jumpToStart();

  for (int i = 0; i < nsubcodes; ++i) {
    uint64_t decoded = code_manager.decode();
    ASSERT_EQ(decoded, testing_vector[i] & mask);
  }
}

} // namespace flatnav::tests