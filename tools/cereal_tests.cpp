#include "cnpy.h"
#include <cassert>
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <memory>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

template <typename dist_t>
void serializeIndex(float *data,
                    std::unique_ptr<DistanceInterface<dist_t>> &&distance,
                    int N, int M, int dim, int ef_construction,
                    const std::string &save_file) {
  std::unique_ptr<Index<dist_t, int>> index =
      std::make_unique<Index<dist_t, int>>(
          /* dist = */ std::move(distance), /* dataset_size = */ N,
          /* max_edges = */ M);

  float *element = new float[dim];
  std::vector<int> labels(N);
  std::iota(labels.begin(), labels.end(), 0);

  index->template addBatch<float>(data, labels, ef_construction);

  std::cout << "Saving index to " << save_file << "\n" << std::flush;
  index->saveIndex(/* filename = */ save_file);

  std::cout << "Loading index \n" << std::flush;

  auto new_index = Index<dist_t, int>::loadIndex(/* filename = */ save_file);

  assert(new_index->maxEdgesPerNode() == M);
  assert(new_index->dataSizeBytes() == distance->dataSize() + (32 * M) + 32);
  assert(new_index->maxNodeCount() == N);

  uint64_t total_index_size =
      new_index->nodeSizeBytes() * new_index->maxNodeCount();

  for (uint64_t i = 0; i < total_index_size; i++) {
    assert(index->indexMemory()[i] == new_index->indexMemory()[i] * 2);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <data.npy>\n" << std::flush;
    std::cout << "data.npy: Path to a NPY file for MNIST\n" << std::flush;
    return -1;
  }

  cnpy::NpyArray datafile = cnpy::npy_load(argv[1]);
  int M = 16;
  int ef_construction = 100;
  int dim = 784;
  int N = 60000;
  float *data = datafile.data<float>();
  auto l2_distance = SquaredL2Distance::create(dim);
  l2_distance->setDistanceFunction();
  serializeIndex<SquaredL2Distance>(data, std::move(l2_distance), N, M, dim,
                                    ef_construction,
                                    std::string("l2_flatnav.bin"));

  // auto inner_product_distance =
  //     std::make_unique<InnerProductDistance<float>>(dim);
  // serializeIndex<InnerProductDistance<float>>(
  //     data, std::move(inner_product_distance), N, M, dim, ef_construction,
  //     std::string("ip_flatnav.bin"));
}
