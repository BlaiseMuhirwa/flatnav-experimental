#include "docs.h"
#include <algorithm>
#include <cstdint>
#include <flatnav/distances/DistanceInterface.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/index/Index.h>
#include <flatnav/util/Datatype.h>
#include <flatnav/util/Multithreading.h>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using flatnav::Index;
using flatnav::distances::DistanceInterface;
using flatnav::distances::InnerProductDistance;
using flatnav::distances::SquaredL2Distance;
using flatnav::util::DataType;
using flatnav::util::for_each_data_type;

namespace py = pybind11;

template <typename Func, typename... Args>
auto cast_and_call(DataType data_type, const py::array &array, Func &&function,
                   Args &&... args) {
  switch (data_type) {
  case DataType::float32:
    return function(
        array.cast<
            py::array_t<float, py::array::c_style | py::array::forcecast>>(),
        std::forward<Args>(args)...);
  case DataType::int8:
    return function(
        array.cast<
            py::array_t<int8_t, py::array::c_style | py::array::forcecast>>(),
        std::forward<Args>(args)...);
  case DataType::uint8:
    return function(
        array.cast<
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast>>(),
        std::forward<Args>(args)...);
  default:
    throw std::invalid_argument("Unsupported data type.");
  }
}

template <typename dist_t, typename label_t>
class PyIndex : public std::enable_shared_from_this<PyIndex<dist_t, label_t>> {

  int _dim;
  label_t _label_id;
  bool _verbose;
  Index<dist_t, label_t> *_index;
  DataType _data_type;

  typedef std::pair<py::array_t<float>, py::array_t<label_t>>
      DistancesLabelsPair;

  // Internal add method that handles templated dispatch
  template <typename data_type>
  void addImpl(const py::array_t<data_type, py::array::c_style |
                                                py::array::forcecast> &data,
               int ef_construction, int num_initializations = 100,
               py::object labels = py::none()) {
    // py::array_t<float, py::array::c_style | py::array::forcecast> means that
    // the functions expects either a Numpy array of floats or a castable type
    // to that type. If the given type can't be casted, pybind11 will throw an
    // error.

    auto num_vectors = data.shape(0);
    auto data_dim = data.shape(1);
    if (data.ndim() != 2 || data_dim != _dim) {
      throw std::invalid_argument(
          "Data has incorrect dimensions. data.ndim() = "
          "`" +
          std::to_string(data.ndim()) + "` and data_dim = `" +
          std::to_string(data_dim) +
          "`. Expected 2D "
          "array with "
          "dimensions "
          "(num_vectors, "
          "dim).");
    }

    if (labels.is_none()) {
      std::vector<label_t> vec_labels(num_vectors);
      std::iota(vec_labels.begin(), vec_labels.end(), 0);

      {
        // Release python GIL while threads are running
        py::gil_scoped_release gil;
        this->_index->template addBatch<data_type>(
            /* data = */ (void *)data.data(0),
            /* labels = */ vec_labels,
            /* ef_construction = */ ef_construction,
            /* num_initializations = */ num_initializations);
      }
      return;
    }

    // Use the provided labels now
    try {
      auto vec_labels = py::cast<std::vector<label_t>>(labels);
      if (vec_labels.size() != num_vectors) {
        throw std::invalid_argument("Incorrect number of labels.");
      }
      {
        // Relase python GIL while threads are running
        py::gil_scoped_release gil;
        this->_index->template addBatch<data_type>(
            /* data = */ (void *)data.data(0), /* labels = */ vec_labels,
            /* ef_construction = */ ef_construction,
            /* num_initializations = */ num_initializations);
      }
    } catch (const py::cast_error &error) {
      throw std::invalid_argument("Invalid labels provided.");
    }
  }

  template <typename data_type>
  DistancesLabelsPair searchSingleImpl(
      const py::array_t<data_type, py::array::c_style | py::array::forcecast>
          &query,
      int K, int ef_search, int num_initializations = 100) {
    if (query.ndim() != 1 || query.shape(0) != _dim) {
      throw std::invalid_argument("Query has incorrect dimensions.");
    }

    std::vector<std::pair<float, label_t>> top_k = this->_index->search(
        /* query = */ (const void *)query.data(0), /* K = */ K,
        /* ef_search = */ ef_search,
        /* num_initializations = */ num_initializations);

    // if (top_k.size() != K) {
    //   throw std::runtime_error(
    //       "Search did not return the expected number of results. Expected " +
    //       std::to_string(K) + " but got " + std::to_string(top_k.size()) +
    //       ".");
    // }

    label_t *labels = new label_t[K];
    float *distances = new float[K];

    for (size_t i = 0; i < K; i++) {
      distances[i] = top_k[i].first;
      labels[i] = top_k[i].second;
    }

    // Allows to transfer ownership to Python
    py::capsule free_labels_when_done(labels,
                                      [](void *ptr) { delete (label_t *)ptr; });

    py::capsule free_distances_when_done(
        distances, [](void *ptr) { delete (float *)ptr; });

    py::array_t<label_t> labels_array = py::array_t<label_t>(
        {K}, {sizeof(label_t)}, labels, free_labels_when_done);

    py::array_t<float> distances_array = py::array_t<float>(
        {K}, {sizeof(float)}, distances, free_distances_when_done);

    return {distances_array, labels_array};
  }

  template <typename data_type>
  DistancesLabelsPair
  searchImpl(const py::array_t<data_type, py::array::c_style |
                                              py::array::forcecast> &queries,
             int K, int ef_search, int num_initializations = 100) {
    size_t num_queries = queries.shape(0);
    size_t queries_dim = queries.shape(1);

    if (queries.ndim() != 2 || queries_dim != _dim) {
      throw std::invalid_argument("Queries have incorrect dimensions.");
    }

    auto num_threads = _index->getNumThreads();
    label_t *results = new label_t[num_queries * K];
    float *distances = new float[num_queries * K];

    // No need to spawn any threads if we are in a single-threaded environment
    if (num_threads == 1) {
      for (size_t query_index = 0; query_index < num_queries; query_index++) {
        std::vector<std::pair<float, label_t>> top_k = this->_index->search(
            /* query = */ (const void *)queries.data(query_index), /* K = */ K,
            /* ef_search = */ ef_search,
            /* num_initializations = */ num_initializations);

        if (top_k.size() != K) {
          throw std::runtime_error("Search did not return the expected number "
                                   "of results. Expected " +
                                   std::to_string(K) + " but got " +
                                   std::to_string(top_k.size()) + ".");
        }

        for (size_t i = 0; i < top_k.size(); i++) {
          distances[query_index * K + i] = top_k[i].first;
          results[query_index * K + i] = top_k[i].second;
        }
      }
    } else {
      // Parallelize the search
      flatnav::executeInParallel(
          /* start_index = */ 0, /* end_index = */ num_queries,
          /* num_threads = */ num_threads,
          /* function = */ [&](uint32_t row_index) {
            auto *query = (const void *)queries.data(row_index);
            std::vector<std::pair<float, label_t>> top_k = this->_index->search(
                /* query = */ query, /* K = */ K, /* ef_search = */ ef_search,
                /* num_initializations = */ num_initializations);

            for (uint32_t result_id = 0; result_id < K; result_id++) {
              distances[(row_index * K) + result_id] = top_k[result_id].first;
              results[(row_index * K) + result_id] = top_k[result_id].second;
            }
          });
    }

    // Allows to transfer ownership to Python
    py::capsule free_results_when_done(
        results, [](void *ptr) { delete (label_t *)ptr; });
    py::capsule free_distances_when_done(
        distances, [](void *ptr) { delete (float *)ptr; });

    py::array_t<label_t> labels =
        py::array_t<label_t>({num_queries, (size_t)K}, // shape of the array
                             {K * sizeof(label_t), sizeof(label_t)}, // strides
                             results,               // data pointer
                             free_results_when_done // capsule
        );

    py::array_t<float> dists = py::array_t<float>(
        {num_queries, (size_t)K}, {K * sizeof(float), sizeof(float)}, distances,
        free_distances_when_done);

    return {labels, dists};
  }

public:
  explicit PyIndex(std::unique_ptr<Index<dist_t, label_t>> index)
      : _dim(index->dataDimension()), _label_id(0), _verbose(false),
        _index(index.release()) {

    if (_verbose) {
      _index->getIndexSummary();
    }
  }

  PyIndex(std::unique_ptr<DistanceInterface<dist_t>> &&distance,
          DataType data_type, int dataset_size, int max_edges_per_node,
          bool verbose = false, bool collect_stats = false)
      : _dim(distance->dimension()), _label_id(0), _verbose(verbose),
        _index(new Index<dist_t, label_t>(
            /* dist = */ std::move(distance),
            /* dataset_size = */ dataset_size,
            /* max_edges_per_node = */ max_edges_per_node,
            /* collect_stats = */ collect_stats)) {

    _data_type = data_type;

    if (_verbose) {
      uint64_t total_index_memory = _index->getTotalIndexMemory();
      uint64_t visited_set_allocated_memory =
          _index->visitedSetPoolAllocatedMemory();
      uint64_t mutexes_allocated_memory = _index->mutexesAllocatedMemory();

      auto total_memory = total_index_memory + visited_set_allocated_memory +
                          mutexes_allocated_memory;

      std::cout << "Total allocated index memory: "
                << (float)(total_memory / 1e9) << " GB \n"
                << std::flush;
      std::cout << "[WARN]: More memory might be allocated due to visited sets "
                   "in multi-threaded environments.\n"
                << std::flush;
      _index->getIndexSummary();
    }
  }

  Index<dist_t, label_t> *getIndex() { return _index; }

  ~PyIndex() { delete _index; }

  inline const std::unordered_map<uint32_t, uint32_t> getNodeAccessCounts() {
    return _index->getNodeAccessCounts();
  }

  uint64_t getQueryDistanceComputations() const {
    auto distance_computations = _index->distanceComputations();
    _index->resetStats();
    return distance_computations;
  }

  void buildGraphLinks(const std::string &mtx_filename) {
    _index->buildGraphLinks(/* mtx_filename = */ mtx_filename);
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    return _index->getGraphOutdegreeTable();
  }

  uint32_t getMaxEdgesPerNode() { return _index->maxEdgesPerNode(); }

  void reorder(const std::vector<std::string> &strategies) {
    // validate the given strategies
    for (auto &strategy : strategies) {
      auto alg = strategy;
      std::transform(alg.begin(), alg.end(), alg.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      if (alg != "gorder" && alg != "rcm") {
        throw std::invalid_argument(
            "`" + strategy +
            "` is not a supported graph re-ordering strategy.");
      }
    }
    _index->doGraphReordering(strategies);
  }

  void setNumThreads(uint32_t num_threads) {
    _index->setNumThreads(num_threads);
  }

  uint32_t getNumThreads() { return _index->getNumThreads(); }

  void save(const std::string &filename) {
    _index->saveIndex(/* filename = */ filename);
  }

  static std::shared_ptr<PyIndex<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    auto index = Index<dist_t, label_t>::loadIndex(/* filename = */ filename);
    return std::make_shared<PyIndex<dist_t, label_t>>(std::move(index));
  }

  std::shared_ptr<PyIndex<dist_t, label_t>> allocateNodes(
      const py::array_t<float, py::array::c_style | py::array::forcecast>
          &data) {
    auto num_vectors = data.shape(0);
    auto data_dim = data.shape(1);
    if (data.ndim() != 2 || data_dim != _dim) {
      throw std::invalid_argument("Data has incorrect dimensions.");
    }
    for (size_t vec_index = 0; vec_index < num_vectors; vec_index++) {
      uint32_t new_node_id;

      this->_index->allocateNode(/* data = */ (void *)data.data(vec_index),
                                 /* label = */ _label_id,
                                 /* new_node_id = */ new_node_id);
      _label_id++;
    }
    return this->shared_from_this();
  }

  void add(const py::array &data, int ef_construction, int num_initializations,
           py::object labels = py::none()) {
    cast_and_call(
        _data_type, data,
        [this](auto &&casted_data, int ef, int num_init, py::object lbls) {
          this->addImpl(std::forward<decltype(casted_data)>(casted_data), ef,
                        num_init, lbls);
        },
        ef_construction, num_initializations, labels);
  }

  DistancesLabelsPair search(const py::array &queries, int K, int ef_search,
                             int num_initializations) {
    return cast_and_call(
        _data_type, queries,
        [this](auto &&casted_queries, int k, int ef, int num_init) {
          return this->searchImpl(
              std::forward<decltype(casted_queries)>(casted_queries), k, ef,
              num_init);
        },
        K, ef_search, num_initializations);
  }

  DistancesLabelsPair searchSingle(const py::array &query, int K, int ef_search,
                                   int num_initializations) {
    return cast_and_call(
        _data_type, query,
        [this](auto &&casted_query, int k, int ef, int num_init) {
          return this->searchSingleImpl(
              std::forward<decltype(casted_query)>(casted_query), k, ef,
              num_init);
        },
        K, ef_search, num_initializations);
  }
};

template <typename dist_t> struct IndexSpecialization;

template <> struct IndexSpecialization<SquaredL2Distance<DataType::float32>> {
  using type = PyIndex<SquaredL2Distance<DataType::float32>, int>;
  static constexpr char *name = "IndexL2Float";
};

template <> struct IndexSpecialization<SquaredL2Distance<DataType::uint8>> {
  using type = PyIndex<SquaredL2Distance<DataType::uint8>, int>;
  static constexpr char *name = "IndexL2Uint8";
};

template <> struct IndexSpecialization<SquaredL2Distance<DataType::int8>> {
  using type = PyIndex<SquaredL2Distance<DataType::int8>, int>;
  static constexpr char *name = "IndexL2Int8";
};

template <>
struct IndexSpecialization<InnerProductDistance<DataType::float32>> {
  using type = PyIndex<InnerProductDistance<DataType::float32>, int>;
  static constexpr char *name = "IndexIPFloat";
};

template <> struct IndexSpecialization<InnerProductDistance<DataType::uint8>> {
  using type = PyIndex<InnerProductDistance<DataType::uint8>, int>;
  static constexpr char *name = "IndexIPUint8";
};

template <> struct IndexSpecialization<InnerProductDistance<DataType::int8>> {
  using type = PyIndex<InnerProductDistance<DataType::int8>, int>;
  static constexpr char *name = "IndexIPInt8";
};

void validateDistanceType(const std::string &distance_type) {
  auto dist_type = distance_type;
  std::transform(dist_type.begin(), dist_type.end(), dist_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (dist_type != "l2" && dist_type != "angular") {
    throw std::invalid_argument("Invalid distance type: `" + dist_type +
                                "` during index construction. Valid options "
                                "include `l2` and `angular`.");
  }
}

template <DataType data_type, typename... Args>
py::object createIndex(const std::string &distance_type, int dim,
                       Args &&... args) {
  validateDistanceType(distance_type);

  if (distance_type == "l2") {
    auto distance = SquaredL2Distance<data_type>::create(dim);
    auto index = std::make_shared<PyIndex<SquaredL2Distance<data_type>, int>>(
        std::move(distance), data_type, std::forward<Args>(args)...);
    return py::cast(index);
  }

  auto distance = InnerProductDistance<data_type>::create(dim);
  auto index = std::make_shared<PyIndex<InnerProductDistance<data_type>, int>>(
      std::move(distance), data_type, std::forward<Args>(args)...);
  return py::cast(index);
}

template <typename dist_t, typename label_t>
void bindSpecialization(py::module_ &index_submodule) {
  using IndexType = typename IndexSpecialization<dist_t>::type;
  auto index_class = py::class_<IndexType, std::shared_ptr<IndexType>>(
      index_submodule, IndexSpecialization<dist_t>::name);

  index_class
      .def(
          "add",
          [](IndexType &index, const py::array &data, int ef_construction,
             int num_initializations = 100, py::object labels = py::none()) {
            index.add(data, ef_construction, num_initializations, labels);
          },
          py::arg("data"), py::arg("ef_construction"),
          py::arg("num_initializations") = 100, py::arg("labels") = py::none(),
          ADD_DOCSTRING)
      .def(
          "allocate_nodes",
          [](IndexType &index,
             const py::array_t<float, py::array::c_style | py::array::forcecast>
                 &data) { return index.allocateNodes(data); },
          py::arg("data"), ALLOCATE_NODES_DOCSTRING)
      .def(
          "search_single",
          [](IndexType &index, const py::array &query, int K, int ef_search,
             int num_initializations = 100) {
            return index.searchSingle(query, K, ef_search, num_initializations);
          },
          py::arg("query"), py::arg("K"), py::arg("ef_search"),
          py::arg("num_initializations") = 100, SEARCH_SINGLE_DOCSTRING)
      .def(
          "search",
          [](IndexType &index, const py::array &queries, int K, int ef_search,
             int num_initializations = 100) {
            return index.search(queries, K, ef_search, num_initializations);
          },
          py::arg("queries"), py::arg("K"), py::arg("ef_search"),
          py::arg("num_initializations") = 100, SEARCH_DOCSTRING)
      .def("get_query_distance_computations",
           &IndexType::getQueryDistanceComputations,
           GET_QUERY_DISTANCE_COMPUTATIONS_DOCSTRING)
      .def("save", &IndexType::save, py::arg("filename"), SAVE_DOCSTRING)
      .def("get_node_access_counts", &IndexType::getNodeAccessCounts,"")
      .def("build_graph_links", &IndexType::buildGraphLinks,
           py::arg("mtx_filename"), BUILD_GRAPH_LINKS_DOCSTRING)
      .def("get_graph_outdegree_table", &IndexType::getGraphOutdegreeTable,
           GET_GRAPH_OUTDEGREE_TABLE_DOCSTRING)
      .def("reorder", &IndexType::reorder, py::arg("strategies"),
           REORDER_DOCSTRING)
      .def("set_num_threads", &IndexType::setNumThreads, py::arg("num_threads"),
           SET_NUM_THREADS_DOCSTRING)
      .def_static("load_index", &IndexType::loadIndex, py::arg("filename"),
                  LOAD_INDEX_DOCSTRING)
      .def_property_readonly("max_edges_per_node",
                             &IndexType::getMaxEdgesPerNode)
      .def_property_readonly("num_threads", &IndexType::getNumThreads,
                             NUM_THREADS_DOCSTRING);
}

void defineIndexSubmodule(py::module_ &index_submodule) {
  bindSpecialization<SquaredL2Distance<DataType::float32>, int>(
      index_submodule);
  bindSpecialization<SquaredL2Distance<DataType::int8>, int>(index_submodule);
  bindSpecialization<SquaredL2Distance<DataType::uint8>, int>(index_submodule);
  bindSpecialization<InnerProductDistance<DataType::float32>, int>(
      index_submodule);
  bindSpecialization<InnerProductDistance<DataType::int8>, int>(
      index_submodule);
  bindSpecialization<InnerProductDistance<DataType::uint8>, int>(
      index_submodule);

  index_submodule.def(
      "create",
      [](const std::string &distance_type, int dim, int dataset_size,
         int max_edges_per_node, DataType index_data_type, bool verbose = false,
         bool collect_stats = false) {
        switch (index_data_type) {
        case DataType::float32:
          return createIndex<DataType::float32>(
              distance_type, dim, dataset_size, max_edges_per_node, verbose,
              collect_stats);
        case DataType::int8:
          return createIndex<DataType::int8>(distance_type, dim, dataset_size,
                                             max_edges_per_node, verbose,
                                             collect_stats);
        case DataType::uint8:
          return createIndex<DataType::uint8>(distance_type, dim, dataset_size,
                                              max_edges_per_node, verbose,
                                              collect_stats);
        default:
          throw std::runtime_error("Unsupported data type");
        }
      },
      py::arg("distance_type"), py::arg("dim"), py::arg("dataset_size"),
      py::arg("max_edges_per_node"),
      py::arg("index_data_type") = DataType::float32,
      py::arg("verbose") = false, py::arg("collect_stats") = false,
      CONSTRUCTOR_DOCSTRING);
}

void defineDatatypeEnums(py::module_ &module) {
  // More enums are available, but these are the only ones that we support
  // for index construction.
  py::enum_<DataType>(module, "DataType")
      .value(flatnav::util::name(DataType::float32), DataType::float32)
      .value(flatnav::util::name(DataType::int8), DataType::int8)
      .value(flatnav::util::name(DataType::uint8), DataType::uint8)
      .export_values();
}

void defineDistanceEnums(py::module_ &module) {
  py::enum_<flatnav::distances::MetricType>(module, "MetricType")
      .value("L2", flatnav::distances::MetricType::L2)
      .value("IP", flatnav::distances::MetricType::IP)
      .export_values();
}

PYBIND11_MODULE(flatnav, module) {
  auto data_type_submodule = module.def_submodule("data_type");
  defineDatatypeEnums(data_type_submodule);

  auto index_submodule = module.def_submodule("index");
  defineIndexSubmodule(index_submodule);
  defineDistanceEnums(module);
}