#include <chrono>
#include <cmath>
#include <flatnav/index.h>
#include <flatnav/distances/inner_product_distance.h>
#include <flatnav/distances/squared_l2_distance.h>
#include <fstream>
#include <iostream>
#include <quantization/product_quantization.h>
#include <random>
#include <utility>
#include <vector>

#include "cnpy.h"
#include <algorithm>
#include <sstream>
#include <string>

using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;
using flatnav::quantization::ProductQuantizer;

template <typename dist_t>
void run(float *queries, int *gtruth, const std::string &index_filename,
         const std::vector<int> &ef_searches, int K, int num_queries,
         int num_gtruth, int dim, bool reorder = true) {

  std::unique_ptr<Index<dist_t, int>> index =
      Index<dist_t, int>::loadIndex(index_filename);

  std::cout << "[INFO] Index loaded" << std::endl;
  index->getIndexSummary();

  if (reorder) {
    std::clog << "[INFO] Gorder Reordering: " << std::endl;
    auto start_r = std::chrono::high_resolution_clock::now();
    index->reorderGOrder();
    auto stop_r = std::chrono::high_resolution_clock::now();
    auto duration_r =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_r - start_r);
    std::clog << "Reordering time: " << (float)(duration_r.count()) / (1000.0)
              << " seconds" << std::endl;
  }

  for (const auto &ef_search : ef_searches) {
    double mean_recall = 0;

    auto start_q = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_queries; i++) {
      float *q = queries + dim * i;
      int *g = gtruth + num_gtruth * i;

      std::vector<std::pair<float, int>> result =
          index->search(q, K, ef_search);

      double recall = 0;
      for (int j = 0; j < K; j++) {
        for (int l = 0; l < K; l++) {
          if (result[j].second == g[l]) {
            recall = recall + 1;
          }
        }
      }
      recall = recall / K;
      mean_recall = mean_recall + recall;
    }
    auto stop_q = std::chrono::high_resolution_clock::now();
    auto duration_q =
        std::chrono::duration_cast<std::chrono::milliseconds>(stop_q - start_q);
    std::cout << "[INFO] Mean Recall: " << mean_recall / num_queries
              << ", Duration:" << (float)(duration_q.count()) / num_queries
              << " milliseconds" << std::endl;
  }
}

int main(int argc, char **argv) {

  if (argc < 9) {
    std::clog << "Usage: " << std::endl;
    std::clog << "query <space> <index> <queries> <gtruth> <ef_search> <k> "
                 "<Reorder ID> <Quantized>"
              << std::endl;
    std::clog << "\t <data> <queries> <gtruth>: .npy files (float, float, int) "
                 "from ann-benchmarks"
              << std::endl;
    std::clog << "\t <M>: int number of links" << std::endl;
    std::clog << "\t <ef_construction>: int " << std::endl;
    std::clog << "\t <ef_search>: int,int,int,int...,int " << std::endl;
    std::clog << "\t <k>: number of neighbors " << std::endl;
    std::clog << "\t <Reorder ID>: 0 for no reordering, 1 for reordering"
              << std::endl;
    std::clog << "\t <Quantized>: 0 for no quantization, 1 for quantization"
              << std::endl;
    return -1;
  }

  int space_ID = std::stoi(argv[1]);
  std::string indexfilename(argv[2]);

  std::vector<int> ef_searches;
  std::stringstream ss(argv[5]);
  int element = 0;
  while (ss >> element) {
    ef_searches.push_back(element);
    if (ss.peek() == ',')
      ss.ignore();
  }
  int k = std::stoi(argv[6]);
  int reorder_ID = std::stoi(argv[7]);
  bool quantized = std::stoi(argv[8]) ? true : false;

  bool reorder = reorder_ID ? true : false;

  cnpy::NpyArray queryfile = cnpy::npy_load(argv[3]);
  cnpy::NpyArray truthfile = cnpy::npy_load(argv[4]);
  if ((queryfile.shape.size() != 2) || (truthfile.shape.size() != 2)) {
    return -1;
  }

  int num_queries = queryfile.shape[0];
  int dim = queryfile.shape[1];
  int n_gt = truthfile.shape[1];
  if (k > n_gt) {
    std::cerr
        << "K is larger than the number of precomputed ground truth neighbors"
        << std::endl;
    return -1;
  }

  std::clog << "Loading " << num_queries << " queries" << std::endl;
  float *queries = queryfile.data<float>();
  std::clog << "Loading " << num_queries
            << " ground truth results with k = " << k << std::endl;
  int *gtruth = truthfile.data<int>();

  if (quantized) {
    run<ProductQuantizer>(/* queries = */ queries, /* gtruth = */
                          gtruth,
                          /* index_filename = */ indexfilename,
                          /* ef_searches = */ ef_searches, /* K = */ k,
                          /* num_queries = */ num_queries,
                          /* num_gtruth = */ n_gt, /* dim = */ dim,
                          /* reorder = */ reorder);
  } else if (space_ID == 0) {
    run<SquaredL2Distance>(/* queries = */ queries, /* gtruth = */ gtruth,
                           /* index_filename = */ indexfilename,
                           /* ef_searches = */ ef_searches, /* K = */ k,
                           /* num_queries = */ num_queries,
                           /* num_gtruth = */ n_gt, /* dim = */ dim,
                           /* reorder = */ reorder);

  } else if (space_ID == 1) {
    run<InnerProductDistance>(/* queries = */ queries, /* gtruth = */
                              gtruth,
                              /* index_filename = */ indexfilename,
                              /* ef_searches = */ ef_searches, /* K = */ k,
                              /* num_queries = */ num_queries,
                              /* num_gtruth = */ n_gt, /* dim = */ dim,
                              /* reorder = */ reorder);

  } else {
    throw std::invalid_argument("Invalid space ID. Valid IDs are 0 and 1.");
  }

  return 0;
}