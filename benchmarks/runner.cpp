#include "../flatnav/Index.h"
#include "../flatnav/distances/InnerProductDistance.h"
#include "../flatnav/distances/SquaredL2Distance.h"
#include "cnpy.h"
#include "config_parser.h"
#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cmath>
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
void buildIndex(std::shared_ptr<Index<dist_t, int>> index, float *data, int N,
                int M, int dim, int ef_construction) {
  for (int label = 0; label < N; label++) {
    float *element = data + (dim * label);
    index->add(/* data = */ (void *)element, /* label = */ label,
               /* ef_construction */ ef_construction);
  }
}

// Assumes we have a file named config.toml file in the current
// directory with the desired benchmarks to run
static const char *CONFIG = "./benchmarks/config.toml";

struct BenchmarkFixture : public benchmark::Fixture {
  std::shared_ptr<Index<SquaredL2Distance, int>> _index;
  int _N, _M, _dim, _ef_construction;
  float *_data;

  void prepareDataset() {

    auto benchmarks = parseBenchmarks(/* filename = */ CONFIG);

    cnpy::NpyArray datafile = cnpy::npy_load(benchmarks[0].name);

    if (datafile.shape.size() != 2) {
      throw std::runtime_error("Invalid benchmark filename");
    }
    _data = datafile.data<float>();
    _N = benchmarks[0].dataset_size;
    _M = benchmarks[0].max_edges;
    _dim = benchmarks[0].dim;
    _ef_construction = benchmarks[0].ef_construction;
  }

  void SetUp(benchmark::State &state) override {

    prepareDataset();
    auto distance = std::make_unique<SquaredL2Distance>(_dim);
    _index = std::make_shared<Index<SquaredL2Distance, int>>(
        /* dist = */ std::move(distance), /* dataset_size = */ _N,
        /* max_edges = */ _M);
  }

  void TearDown(benchmark::State &state) override { _index.reset(); }
};

BENCHMARK_DEFINE_F(BenchmarkFixture, Test)(benchmark::State &state) {
  // This will run the actual benchmark
  for (auto _ : state) {
    buildIndex<SquaredL2Distance>(_index, _data, _N, _M, _dim,
                                  _ef_construction);
  }
}

BENCHMARK_REGISTER_F(BenchmarkFixture, Test);

BENCHMARK_MAIN();
