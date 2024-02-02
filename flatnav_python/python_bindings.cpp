#include <algorithm>
#include <flatnav/DistanceInterface.h>
#include <flatnav/Index.h>
#include <flatnav/distances/InnerProductDistance.h>
#include <flatnav/distances/SquaredL2Distance.h>
#include <flatnav/util/ParallelConstructs.h>
#include <iostream>
#include <memory>
#include <ostream>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using flatnav::DistanceInterface;
using flatnav::Index;
using flatnav::InnerProductDistance;
using flatnav::SquaredL2Distance;

namespace py = pybind11;

template <typename dist_t, typename label_t>
class PyIndex : public std::enable_shared_from_this<PyIndex<dist_t, label_t>> {
  const uint32_t NUM_LOG_STEPS = 10000;

private:
  int _dim;
  label_t _label_id;
  bool _verbose;
  Index<dist_t, label_t> *_index;

public:
  typedef std::pair<py::array_t<float>, py::array_t<label_t>>
      DistancesLabelsPair;

  typedef std::pair<py::array_t<label_t>, py::array_t<float>>
      LabelsDistancesPair;

  explicit PyIndex(std::unique_ptr<Index<dist_t, label_t>> index)
      : _dim(index->dataDimension()), _label_id(0), _verbose(false),
        _index(index.get()) {

    if (_verbose) {
      _index->getIndexSummary();
    }
  }

  PyIndex(std::shared_ptr<DistanceInterface<dist_t>> distance, int dataset_size,
          int max_edges_per_node, bool verbose = false)
      : _dim(distance->dimension()), _label_id(0), _verbose(verbose),
        _index(new Index<dist_t, label_t>(
            /* dist = */ std::move(distance),
            /* dataset_size = */ dataset_size,
            /* max_edges_per_node = */ max_edges_per_node)) {

    if (_verbose) {
      _index->getIndexSummary();
    }
  }

  PyIndex(std::shared_ptr<DistanceInterface<dist_t>> distance,
          const std::string &mtx_filename, bool verbose = false)
      : _label_id(0), _verbose(verbose),
        _index(new Index<dist_t, label_t>(/* dist = */ std::move(distance),
                                          /* mtx_filename = */ mtx_filename)) {
    _dim = _index->dataDimension();
  }

  Index<dist_t, label_t> *getIndex() { return _index; }

  ~PyIndex() { delete _index; }

  static std::unique_ptr<PyIndex<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    auto index = Index<dist_t, label_t>::loadIndex(/* filename = */ filename);
    return std::make_unique<PyIndex<dist_t, label_t>>(std::move(index));
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

  void
  add(const py::array_t<float, py::array::c_style | py::array::forcecast> &data,
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
        this->_index->addBatch(
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
        this->_index->addBatch(
            /* data = */ (void *)data.data(0), /* labels = */ vec_labels,
            /* ef_construction = */ ef_construction,
            /* num_initializations = */ num_initializations);
      }
    } catch (const py::cast_error &error) {
      throw std::invalid_argument("Invalid labels provided.");
    }
  }

  LabelsDistancesPair
  search(const py::array_t<float, py::array::c_style | py::array::forcecast>
             queries,
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
};

using L2FlatNavIndex = PyIndex<SquaredL2Distance, int>;
using InnerProductFlatNavIndex = PyIndex<InnerProductDistance, int>;

template <typename IndexType>
void bindIndexMethods(
    py::class_<IndexType, std::shared_ptr<IndexType>> &index_class) {
  index_class
      .def(
          "save",
          [](IndexType &index_type, const std::string &filename) {
            auto index = index_type.getIndex();
            index->saveIndex(/* filename = */ filename);
          },
          py::arg("filename"),
          "Save a FlatNav index at the given file location.")
      .def_static("load", &IndexType::loadIndex, py::arg("filename"),
                  "Load a FlatNav index from a given file location")
      .def("add", &IndexType::add, py::arg("data"), py::arg("ef_construction"),
           py::arg("num_initializations") = 100, py::arg("labels") = py::none(),
           "Add vectors(data) to the index with the given `ef_construction` "
           "parameter and optional labels. `ef_construction` determines how "
           "many "
           "vertices are visited while inserting every vector in the "
           "underlying graph structure.")
      .def("allocate_nodes", &IndexType::allocateNodes, py::arg("data"),
           "Allocate nodes in the underlying graph structure for the given "
           "data. Unlike the add method, this method does not construct the "
           "edge connectivity. It only allocates memory for each node in the "
           "grpah. When using this method, you should invoke "
           "`build_graph_links` explicity. NOTE: In most cases you should not "
           "need to use this method.")
      .def("search", &IndexType::search, py::arg("queries"), py::arg("K"),
           py::arg("ef_search"), py::arg("num_initializations") = 100,
           "Return top `K` closest data points for every query in the "
           "provided `queries`. The results are returned as a Tuple of "
           "distances and label ID's. The `ef_search` parameter determines "
           "how "
           "many neighbors are visited while finding the closest neighbors "
           "for every query.")
      .def(
          "get_graph_outdegree_table",
          [](IndexType &index_type) -> std::vector<std::vector<uint32_t>> {
            auto index = index_type.getIndex();
            return index->getGraphOutdegreeTable();
          },
          "Returns the outdegree table (adjacency list) representation of "
          "the "
          "underlying graph.")
      .def(
          "build_graph_links",
          [](IndexType &index_type) {
            auto index = index_type.getIndex();
            index->buildGraphLinks();
          },
          "Construct the edge connectivity of the underlying graph. This "
          "method "
          "should be invoked after allocating nodes using the "
          "`allocate_nodes` "
          "method.")
      .def(
          "reorder",
          [](IndexType &index_type,
             const std::vector<std::string> &strategies) {
            auto index = index_type.getIndex();
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
            for (auto &strategy : strategies) {
              auto alg = strategy;
              std::transform(alg.begin(), alg.end(), alg.begin(),
                             [](unsigned char c) { return std::tolower(c); });
              if (alg == "gorder") {
                index->reorderGOrder();
              } else if (alg == "rcm") {
                index->reorderRCM();
              }
            }
          },
          py::arg("strategies"),
          "Perform graph re-ordering based on the given sequence of "
          "re-ordering strategies. "
          "Supported re-ordering strategies include `gorder` and `rcm`.")
      .def(
          "set_num_threads",
          [](IndexType &index_type, uint32_t num_threads) {
            auto *index = index_type.getIndex();
            index->setNumThreads(num_threads);
          },
          py::arg("num_threads"),
          "Set the number of threads to use for constructing the graph and/or "
          "performing KNN search.")
      .def_property_readonly(
          "num_threads",
          [](IndexType &index_type) {
            auto *index = index_type.getIndex();
            return index->getNumThreads();
          },
          "Returns the number of threads used for "
          "constructing the graph and/or performing KNN "
          "search.")
      .def_property_readonly(
          "max_edges_per_node",
          [](IndexType &index_type) {
            return index_type.getIndex()->maxEdgesPerNode();
          },
          "Maximum number of edges(links) per node in the underlying NSW "
          "graph "
          "data structure.");
}

template <typename... Args>
py::object createIndex(const std::string &distance_type, int dim,
                       Args &&...args) {
  auto dist_type = distance_type;
  std::transform(dist_type.begin(), dist_type.end(), dist_type.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  if (dist_type == "l2") {
    auto distance = std::make_shared<SquaredL2Distance>(/* dim = */ dim);
    return py::cast(std::make_shared<L2FlatNavIndex>(
        std::move(distance), std::forward<Args>(args)...));
  } else if (dist_type == "angular") {
    auto distance = std::make_shared<InnerProductDistance>(/* dim = */ dim);
    return py::cast(std::make_shared<InnerProductFlatNavIndex>(
        std::move(distance), std::forward<Args>(args)...));
  }
  throw std::invalid_argument("Invalid distance type: `" + dist_type +
                              "` during index construction. Valid options "
                              "include `l2` and `angular`.");
}

void defineIndexSubmodule(py::module_ &index_submodule) {
  index_submodule.def(
      "index_factory",
      [](const std::string &distance_type, int dim, int dataset_size,
         int max_edges_per_node, bool verbose = false) {
        return createIndex(distance_type, dim, dataset_size, max_edges_per_node,
                           verbose);
      },
      py::arg("distance_type"), py::arg("dim"), py::arg("dataset_size"),
      py::arg("max_edges_per_node"), py::arg("verbose") = false,
      "Creates a FlatNav index given the corresponding "
      "parameters. The `distance_type` argument determines the "
      "kind of index created (either L2Index or IPIndex)");

  index_submodule.def(
      "index_factory",
      [](const std::string &distance_type, int dim,
         const std::string &mtx_filename, bool verbose = false) {
        return createIndex(distance_type, dim, mtx_filename, verbose);
      },
      py::arg("distance_type"), py::arg("dim"), py::arg("mtx_filename"),
      py::arg("verbose") = false,
      "Creates a FlatNav index given the corresponding "
      "parameters. The `distance_type` argument determines the "
      "kind of index created (either L2Index or IPIndex). The "
      "mtx_filename argument is the path to a Matrix Market "
      "file representing the underlying graph's edge connectivity.");

  py::class_<L2FlatNavIndex, std::shared_ptr<L2FlatNavIndex>> l2_index_class(
      index_submodule, "L2Index");
  bindIndexMethods(l2_index_class);

  py::class_<InnerProductFlatNavIndex,
             std::shared_ptr<InnerProductFlatNavIndex>>
      ip_index_class(index_submodule, "IPIndex");
  bindIndexMethods(ip_index_class);
}

PYBIND11_MODULE(flatnav, module) {
  auto index_submodule = module.def_submodule("index");

  defineIndexSubmodule(index_submodule);
}