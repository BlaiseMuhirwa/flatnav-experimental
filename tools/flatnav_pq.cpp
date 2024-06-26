#include "cnpy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <flatnav/index.h>
#include <flatnav/distances/inner_product_distance.h>
#include <flatnav/distances/squared_l2_distance.h>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

template <typename dist_t>
void run(float *data,
         std::unique_ptr<flatnav::DistanceInterface<dist_t>> &&distance, int N,
         int M, int dim, int ef_construction, const std::string &save_file) {
  auto index = new Index<dist_t, int>(
      /* dist = */ std::move(distance), /* dataset_size = */ N,
      /* max_edges = */ M);

  auto start = std::chrono::high_resolution_clock::now();

  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
    if (label % 100000 == 0)
      std::clog << "." << std::flush;
  }
  std::clog << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::clog << "Build time: " << (float)duration.count() << " milliseconds"
            << std::endl;

  std::clog << "Saving index to: " << save_file << std::endl;
  index->saveIndex(/* filename = */ save_file);

  delete index;
}

std::vector<uint8_t> quantize(float *vectors, uint64_t vec_count, uint32_t dim,
                              uint32_t M, uint32_t nbits) {
  auto distance = std::make_unique<SquaredL2Distance>(dim);
  ProductQuantizer<SquaredL2Distance> pq(/* dist = */ std::move(distance),
                                         /* dim = */ dim, /* M = */ M,
                                         /* nbits = */ nbits);

  pq.train(/* vectors = */ vectors, /* num_vectors */ vec_count);

  auto code_size = pq.getCodeSize();
  std::cout << "[INFO] Code size: " << code_size << std::endl;

  std::vector<uint8_t> codes(code_size * vec_count);
  pq.computePQCodes(/* vectors = */ vectors,
                    /* codes = */ codes.data(), /* n = */ vec_count);

  std::cout << "[INFO] Saving codes to: " << "codes.bin" << std::endl;
  std::ofstream stream("codes.bin");
  stream.write((char *)codes.data(), codes.size());

  return codes;
}

int main(int argc, char **argv) {
  // Quantize
  const bool quantize = false;

  // Euclidean metric by default
  const int metric_id = 0;
  // dimension
  const int dim = 784;
  // max edges
  const int max_edges = 16;
  // ef_construction
  const int ef_construction = 100;
  // dataset_size
  const int dataset_size = 60000;

  // datafile
  const char *filename = "mnnist-784-euclidean.train.npy";
  cnpy::NpyArray datafile = cnpy::npy_load(filename);

  assert(datafile.shape.size() == 2);
  assert(datafile.shape[0] == dataset_size);
  assert(datafile.shape[1] == dim);

  std::clog << "Loading " << dim
            << "-dimensional dataset with N = " << dataset_size << std::endl;
  float *data = datafile.data<float>();

  if (quantize) {
    // NOTE: M here is different from max_edges.
    std::vector<uint8_t> codes =
        quantize(/* vectors = */ data, /* vec_count = */ dataset_size,
                 /* dim = */ dim, /* M = */ 8, /* nbits = */ 8);
  }

  auto distance = std::make_unique<SquaredL2Distance>(dim);
  auto index = std::make_unique<Index<SquaredL2Distance, int>>(
      /* dist = */ std::move(distance), /* dataset_size = */ dataset_size,
      /* max_edges = */ max_edges);

  auto start = std::chrono::high_resolution_clock::now();
  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
    if (label % 100000 == 0)
      std::clog << "." << std::flush;
  }
  std::clog << std::endl;

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
  std::clog << "Build time: " << (float)duration.count() << " milliseconds"
            << std::endl;

  if (metric_id == 0) {
    auto distance = std::make_unique<SquaredL2Distance>(dim);
    run<SquaredL2Distance>(
        /* data = */ data,
        /* distance = */ std::move(distance),
        /* N = */ N, /* M = */ M, /* dim = */ dim,
        /* ef_construction = */ ef_construction, /* save_file = */ argv[5]);
  } else if (metric_id == 1) {
    auto distance = std::make_unique<InnerProductDistance>(dim);
    run<InnerProductDistance>(
        /* data = */ data,
        /* distance = */ std::move(distance),
        /* N = */ N, /* M = */ M, dim,
        /* ef_construction = */ ef_construction, /* save_file = */ argv[5]);
  } else {
    throw std::invalid_argument("Provided metric ID " +
                                std::to_string(metric_id) + "is invalid.");
  }

  return 0;
}
