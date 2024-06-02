#include "gtest/gtest.h"
#include <cassert>
#include <cstdio> // for remove
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <random>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

namespace flatnav::testing {

static const uint32_t INDEXED_VECTORS = 10000;
static const uint32_t QUERY_VECTORS = 500;
static const uint32_t VEC_DIM = 500;
static const uint32_t K = 10;
static const uint32_t EF_SEARCH = 50;

// TODO: This is duplicated a couple times. Move it to a common testing
// utils file.
std::vector<float> generateRandomVectors(uint32_t num_vectors, uint32_t dim) {
  std::vector<float> vectors(num_vectors * dim);
  for (uint32_t i = 0; i < num_vectors * dim; i++) {
    vectors[i] = (float)rand() / RAND_MAX;
  }
  return vectors;
}

template <typename dist_t, typename label_t>
void runTest(float *data, std::unique_ptr<DistanceInterface<dist_t>> &&distance,
             int N, int M, int dim, int ef_construction,
             const std::string &save_file) {
  auto data_size = distance->dataSize();

  std::unique_ptr<Index<dist_t, label_t>> index =
      std::make_unique<Index<dist_t, label_t>>(
          /* dist = */ std::move(distance), /* dataset_size = */ N,
          /* max_edges = */ M);

  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);
  index->template addBatch<float>(data, labels, ef_construction);
  index->saveIndex(/* filename = */ save_file);

  auto new_index =
      Index<dist_t, label_t>::loadIndex(/* filename = */ save_file);

  ASSERT_EQ(new_index->maxEdgesPerNode(), M);
  ASSERT_EQ(new_index->dataSizeBytes(), index->dataSizeBytes());
  // 4 bytes for each node_id, 4 bytes for the label
  ASSERT_EQ(new_index->nodeSizeBytes(), data_size + (4 * M) + 4);
  ASSERT_EQ(new_index->maxNodeCount(), N);

  uint64_t total_index_size =
      new_index->nodeSizeBytes() * new_index->maxNodeCount();

  std::vector<float> queries = generateRandomVectors(QUERY_VECTORS, dim);

  for (uint32_t i = 0; i < QUERY_VECTORS; i++) {
    float *q = queries.data() + (dim * i);

    std::vector<std::pair<float, int>> query_result =
        index->search(q, K, EF_SEARCH);

    std::vector<std::pair<float, int>> new_query_result =
        new_index->search(q, K, EF_SEARCH);

    for (uint32_t j = 0; j < K; j++) {
      ASSERT_EQ(query_result[j].first, new_query_result[j].first);
      ASSERT_EQ(query_result[j].second, new_query_result[j].second);
    }
  }
}

TEST(FlatnavSerializationTest, TestL2IndexSerialization) {
  auto vectors = generateRandomVectors(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<SquaredL2Distance>(VEC_DIM);
  std::string save_file = "l2_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<SquaredL2Distance, int>(
      /* data = */ vectors.data(), /* distance */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction = */ ef_construction,
      /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

TEST(FlatnavSerializationTest, TestInnerProductIndexSerialization) {
  auto vectors = generateRandomVectors(INDEXED_VECTORS, VEC_DIM);
  auto distance = std::make_unique<InnerProductDistance>(VEC_DIM);
  std::string save_file = "ip_index.bin";

  uint32_t ef_construction = 100;
  uint32_t M = 16;

  runTest<InnerProductDistance, int>(
      /* data = */ vectors.data(), /* distance = */ std::move(distance),
      /* N = */ INDEXED_VECTORS, /* M = */ M, /* dim = */ VEC_DIM,
      /* ef_construction */ ef_construction, /* save_file = */ save_file);

  EXPECT_EQ(std::remove(save_file.c_str()), 0);
}

} // namespace flatnav::testing