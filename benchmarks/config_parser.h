#pragma once

#include <algorithm>
#include <iostream>
#include <stdint.h>
#include <string>
#include <string_view>
#include <toml++/toml.h>
#include <vector>

/**
 * Defines the standard benchmark config for the ANN-benchmark datasets.
 */
struct Benchmark {
  std::string name;
  uint32_t metric_id;
  uint32_t dim;
  uint32_t max_edges;
  uint32_t ef_construction;
  uint32_t dataset_size;
};

static std::vector<Benchmark> parseBenchmarks(const std::string &filename) {
  std::vector<Benchmark> benchmarks;
  auto config = toml::parse_file(filename);

  try {
    auto benchmark_configs = config["benchmark"].as_array();

    for (auto &benchmark_config : *benchmark_configs) {
      Benchmark bench;

      bench.name =
          benchmark_config.at_path("name").value_exact<std::string>().value();
      bench.metric_id =
          benchmark_config.at_path("metric_id").value<uint32_t>().value();
      bench.dim = benchmark_config.at_path("dim").value<uint32_t>().value();
      bench.max_edges =
          benchmark_config.at_path("max_edges").value<uint32_t>().value();
      bench.ef_construction =
          benchmark_config.at_path("ef_construction").value<uint32_t>().value();
      bench.dataset_size =
          benchmark_config.at_path("dataset_size").value<uint32_t>().value();

      benchmarks.push_back(bench);
    }
  } catch (const toml::parse_error &error) {
    std::cerr << "Toml parsing exception: " << error.what() << "\n";
  }

  return benchmarks;
}