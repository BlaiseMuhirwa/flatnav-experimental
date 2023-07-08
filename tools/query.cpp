#include "cnpy.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <flatnav/Index.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using flatnav::Index;
using flatnav::SquaredL2Distance;

std::shared_ptr<Index<SquaredL2Distance, int>>
buildIndex(float *data, uint32_t dim, uint64_t N, uint32_t max_edges,
           uint32_t ef_construction) {
  auto distance = std::make_unique<SquaredL2Distance>(dim);
  auto index = std::make_shared<Index<SquaredL2Distance, int>>(
      /* dist = */ std::move(distance), /* dataset_size = */ N,
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
  return index;
}

int main(int argc, char **argv) {

  if (argc < 6) {
    std::clog << "Usage: " << std::endl;
    std::clog << "query <data> <space> <queries> <gtruth> <ef_search> <k>";
    std::clog << " [--nq num_queries] [--reorder_id reorder_id] [--ef_profile "
                 "ef_profile] [--num_profile num_profile]"
              << std::endl;
    std::clog << "Positional arguments:" << std::endl;
    std::clog << "\t index: Filename for the training data (float32 index)."
              << std::endl;
    std::clog << "\t space: Integer distance ID: 0 for L2 distance, 1 for "
                 "inner product (angular distance)."
              << std::endl;
    std::clog << "\t queries: Filename for queries (float32 file)."
              << std::endl;
    std::clog << "\t gtruth: Filename for ground truth (int32 file)."
              << std::endl;

    std::clog << "\t k: Number of neighbors to return." << std::endl;

    std::clog << "Optional arguments:" << std::endl;
    std::clog << "\t [--nq num_queries]: (Optional, default 0) Number of "
                 "queries to use. If 0, uses all queries."
              << std::endl;
    std::clog << "\t [--reorder_id reorder_id]: (Optional, default 0) Which "
                 "reordering algorithm to use? 0:none 1:gorder 2:indegsort "
                 "3:outdegsort 4:RCM 5:hubsort 6:hubcluster 7:DBG 8:corder "
                 "91:profiled_gorder 94:profiled_rcm 41:RCM+gorder"
              << std::endl;
    std::clog << "\t [--ef_profile ef_profile]: (Optional, default 100) "
                 "ef_search parameter to use for profiling."
              << std::endl;
    std::clog << "\t [--num_profile num_profile]: (Optional, default 1000) "
                 "Number of queries to use for profiling."
              << std::endl;
    return -1;
  }

  // Optional arguments.
  int num_queries = 0;
  bool reorder = false;
  int reorder_ID = 0;
  int ef_profile = 100;
  int num_profile = 1000;

  for (int i = 0; i < argc; ++i) {
    if (std::strcmp("--nq", argv[i]) == 0) {
      if ((i + 1) < argc) {
        num_queries = std::stoi(argv[i + 1]);
      } else {
        std::cerr << "Invalid argument for optional parameter --nq"
                  << std::endl;
        return -1;
      }
    }
    if (std::strcmp("--reorder_id", argv[i]) == 0) {
      if ((i + 1) < argc) {
        reorder_ID = std::stoi(argv[i + 1]);
      } else {
        std::cerr << "Invalid argument for optional parameter --reorder_id"
                  << std::endl;
        return -1;
      }
    }
    if (std::strcmp("--ef_profile", argv[i]) == 0) {
      if ((i + 1) < argc) {
        ef_profile = std::stoi(argv[i + 1]);
      } else {
        std::cerr << "Invalid argument for optional parameter --ef_profile"
                  << std::endl;
        return -1;
      }
    }
    if (std::strcmp("--num_profile", argv[i]) == 0) {
      if ((i + 1) < argc) {
        num_profile = std::stoi(argv[i + 1]);
      } else {
        std::cerr << "Invalid argument for optional parameter --num_profile"
                  << std::endl;
        return -1;
      }
    }
  }
  // Positional arguments.
  std::string indexfilename(argv[1]); // Index filename.
  int space_ID = std::stoi(argv[2]);  // Space ID for querying.

  // Load queries.
  std::clog << "[INFO] Loading queries." << std::endl;
  cnpy::NpyArray queries_file = cnpy::npy_load(argv[3]);
  float *queries = queries_file.data<float>();

  // Load ground truth.
  std::clog << "[INFO] Loading ground truth." << std::endl;
  cnpy::NpyArray gtruth_file = cnpy::npy_load(argv[4]);
  uint32_t *gtruth = gtruth_file.data<uint32_t>();

  // EF search vector.
  std::vector<int> ef_searches{100, 102};

  // Number of search results.
  int k = std::stoi(argv[6]);

  std::clog << "[INFO] Loading training data." << std::endl;
  cnpy::NpyArray datafile = cnpy::npy_load(argv[1]);
  float *data = datafile.data<float>();

  std::clog << "[INFO] Building index from " << indexfilename << std::endl;

  uint32_t dim = 784;
  auto index = buildIndex(/* data = */ data, /* dim = */ dim, /* N = */ 60000,
                          /* max_edges = */ 16, /* ef_construction = */ 100);

  // Do reordering, if necessary.
  if (num_profile > num_queries) {
    std::clog << "Warning: Number of profiling queries (" << num_profile
              << ") is greater than number of queries (" << num_queries << ")!"
              << std::endl;
    num_profile = num_queries;
  }
  if (reorder) {
    std::clog << "Using GORDER" << std::endl;
    std::clog << "Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index->reorder_gorder();
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reorder time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  } else {
    std::clog << "No reordering" << std::endl;
  }

  int num_gtruth_entries = 100;

  // Now, finally, do the actual search.
  std::cout << "recall, mean_latency_ms" << std::endl;
  for (int &ef_search : ef_searches) {
    double mean_recall = 0;

    auto start_q = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; i++) {
      float *q = queries + dim * i;
      unsigned int *g = gtruth + num_gtruth_entries * i;

      std::vector<std::pair<float, int>> result =
          index->search(q, k, ef_search);

      double recall = 0;
      for (int j = 0; j < k; j++) {
        for (int l = 0; l < k; l++) {
          if (result[j].second == g[l]) {
            recall = recall + 1;
          }
        }
      }
      recall = recall / k;
      mean_recall = mean_recall + recall;
    }
    auto stop_q = std::chrono::high_resolution_clock::now();
    auto duration_q =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_q - start_q);
    std::cout << mean_recall / num_queries << ","
              << (float)(duration_q.count()) / num_queries << std::endl;
  }

  delete[] queries;
  delete[] gtruth;
  delete[] data;

  return 0;
}