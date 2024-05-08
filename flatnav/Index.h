#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cstring>
#include <flatnav/DistanceInterface.h>
#include <flatnav/util/Macros.h>
#include <flatnav/util/ParallelConstructs.h>
#include <flatnav/util/Reordering.h>
#include <flatnav/util/VisitedSetPool.h>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flatnav {

// Define a custom hash function for std::pair<uint32_t, uint32_t>
struct pair_hash {
  template <class T1, class T2>
  std::size_t operator()(const std::pair<T1, T2> &pair) const {
    auto hash1 = std::hash<T1>{}(pair.first);
    auto hash2 = std::hash<T2>{}(pair.second);
    // Combine the two hash values. (This is just one possible way to do it.)
    return hash1 ^ hash2;
  }
};

// dist_t: A distance function implementing DistanceInterface.
// label_t: A fixed-width data type for the label (meta-data) of each point.
template <typename dist_t, typename label_t> class Index {
  typedef std::pair<float, label_t> dist_label_t;
  // internal node numbering scheme. We might need to change this to uint64_t
  typedef uint32_t node_id_t;
  typedef std::pair<float, node_id_t> dist_node_t;

  // NOTE: by default this is a max-heap. We could make this a min-heap
  // by using std::greater, but we want to use the queue as both a max-heap and
  // min-heap depending on the context.
  typedef std::priority_queue<dist_node_t, std::vector<dist_node_t>>
      PriorityQueue;

  // Large (several GB), pre-allocated block of memory.
  char *_index_memory;

  size_t _M;
  // size of one data point (does not support variable-size data, strings)
  size_t _data_size_bytes;
  // Node consists of: ([data] [M links] [data label]). This layout was chosen
  // after benchmarking - it's slightly more cache-efficient than others.
  size_t _node_size_bytes;
  size_t _max_node_count; // Determines size of internal pre-allocated memory
  size_t _cur_num_nodes;
  std::unique_ptr<DistanceInterface<dist_t>> _distance;
  std::mutex _index_data_guard;

  uint32_t _num_threads;

  // Remembers which nodes we've visited, to avoid re-computing distances.
  VisitedSetPool *_visited_set_pool;
  std::vector<std::mutex> _node_links_mutexes;

  bool _collect_stats = false;

  // These are currently only supported for single-threaded search.
  // Trying to use them in multi-threaded setting will result in wird behavior.
  mutable std::atomic<uint64_t> _distance_computations = 0;
  mutable std::atomic<uint64_t> _metric_hops = 0;

  Index(const Index &) = delete;
  Index &operator=(const Index &) = delete;

  // A custom move constructor is needed because the class manages dynamic
  // resources (_index_memory, _visited_set_pool),
  // which require explicit ownership transfer and cleanup to avoid resource
  // leaks or double frees. The default move constructor cannot ensure these
  // resources are safely transferred and the source object is left in a valid
  // state.
  Index(Index &&other) noexcept
      : _index_memory(other._index_memory), _M(other._M),
        _data_size_bytes(other._data_size_bytes),
        _node_size_bytes(other._node_size_bytes),
        _max_node_count(other._max_node_count),
        _cur_num_nodes(other._cur_num_nodes),
        _distance(std::move(other._distance)),
        _index_data_guard(std::move(other._index_data_guard)),
        _num_threads(other._num_threads),
        _visited_set_pool(std::move(other._visited_set_pool)),
        _node_links_mutexes(std::move(other._node_links_mutexes)) {
    other._index_memory = nullptr;
    other._visited_set_pool = nullptr;
  }

  Index &operator=(Index &&other) noexcept {
    if (this != &other) {
      delete[] _index_memory;
      delete _visited_set_pool;

      _index_memory = other._index_memory;
      _M = other._M;
      _data_size_bytes = other._data_size_bytes;
      _node_size_bytes = other._node_size_bytes;
      _max_node_count = other._max_node_count;
      _cur_num_nodes = other._cur_num_nodes;
      _distance = std::move(other._distance);
      _index_data_guard = std::move(other._index_data_guard);
      _num_threads = other._num_threads;
      _visited_set_pool = std::move(other._visited_set_pool);
      _node_links_mutexes = std::move(other._node_links_mutexes);

      other._index_memory = nullptr;
      other._visited_set_pool = nullptr;
    }
    return *this;
  }

  // Tracking metrics for node access patterns. This unordered map is used to
  // record how many times each node is visited during search. The key is the
  // node id and the value is the number of times the node is visited.
  std::unordered_map<uint32_t, uint32_t> _node_access_counts;
  std::mutex _node_access_counts_guard;

  // Tracking metrics for the edge length distribution. This unordered map is
  // used to record the length of each edge in the graph. The key is the hash of
  // the sum of the two node IDs and the value is the length of the edge
  // connecting them.
  std::unordered_map<size_t, float> _edge_length_distribution;

  // Track the number of times each edge is visited during search. This
  // unordered map is used to record how many times each edge is visited during
  // search.
  std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t, pair_hash>
      _edge_access_counts;
  std::mutex _edge_access_counts_guard;

  // Hasher for the node ID's. We add two node IDs and compute their hashes
  // and use the hash as the key in the edge length distribution map.
  std::hash<uint64_t> _hasher;

  // Randomization parameters
  bool _use_random_initialization = false;
  std::mt19937 _generator;
  std::uniform_int_distribution<> _distribution;

  template <typename Archive> void serialize(Archive &archive) {
    archive(_M, _data_size_bytes, _node_size_bytes, _max_node_count,
            _cur_num_nodes, *_distance);

    // Serialize the allocated memory for the index & query.
    archive(
        cereal::binary_data(_index_memory, _node_size_bytes * _max_node_count));
  }

public:
  /**
   * @brief Construct a new Index object for approximate near neighbor search
   *
   * @param dist                A distance metric for the specific index
   * distance. Options include l2(euclidean) and inner product.
   * @param dataset_size        The maximum number of vectors that can be
   * inserted in the index.
   * @param max_edges_per_node  The maximum number of links per node.
   */
  Index(std::unique_ptr<DistanceInterface<dist_t>> dist, int dataset_size,
        int max_edges_per_node, bool collect_stats = false, bool use_random_initialization = false, std::optional<size_t> random_seed = std::nullopt)
      : _M(max_edges_per_node), _max_node_count(dataset_size),
        _cur_num_nodes(0), _distance(std::move(dist)), _num_threads(1),
        _visited_set_pool(new VisitedSetPool(
            /* initial_pool_size = */ 1,
            /* num_elements = */ dataset_size)),
        _node_links_mutexes(dataset_size), _collect_stats(collect_stats), _use_random_initialization(use_random_initialization) {
    
    if (_use_random_initialization) {
      if (!random_seed.has_value()) {
        throw std::invalid_argument(
            "Random seed must be provided if random initialization is used.");
      }
      _generator = std::mt19937(random_seed.value());
      _distribution = std::uniform_int_distribution<>(0, _max_node_count - 1);
    }

    _data_size_bytes = _distance->dataSize();
    _node_size_bytes =
        _data_size_bytes + (sizeof(node_id_t) * _M) + sizeof(label_t);
    size_t index_memory_size = _node_size_bytes * _max_node_count;
    _index_memory = new char[index_memory_size];

  }


  ~Index() {
    delete[] _index_memory;
    delete _visited_set_pool;
  }

  void buildGraphLinks(const std::string &mtx_filename) {
    std::ifstream input_file(mtx_filename);
    if (!input_file.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " +
                               mtx_filename);
    }

    std::string line;
    // Skip the header
    while (std::getline(input_file, line)) {
      if (line[0] != '%')
        break;
    }

    std::istringstream iss(line);
    int num_vertices, num_edges;
    iss >> num_vertices >> num_vertices >> num_edges;

    // check that the number of vertices in the mtx file matches the number of
    // nodes in the index and that the number of edges is equal to the number of
    // links per node.
    if (num_vertices != _max_node_count) {
      throw std::runtime_error("Number of vertices in the mtx file does not "
                               "match the size allocated for the index.");
    }

    if (num_edges != _M) {
      throw std::runtime_error("Number of edges in the mtx file does not match "
                               "the number of links per node.");
    }

    int u, v;
    while (input_file >> u >> v) {
      // Adjust for 1-based indexing in Matrix Market format
      u--;
      v--;
      node_id_t *links = getNodeLinks(u);
      // Now add a directed edge from u to v. We need to check for the first
      // available slot in the links array since there might be other edges
      // added before this one. By definition, a slot is available if and only
      // if it points to the node itself.
      for (size_t i = 0; i < _M; i++) {
        if (links[i] == u) {
          links[i] = v;
          break;
        }
      }
    }

    input_file.close();
  }

  std::vector<std::vector<uint32_t>> getGraphOutdegreeTable() {
    std::vector<std::vector<uint32_t>> outdegree_table(_cur_num_nodes);
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t *links = getNodeLinks(node);
      for (int i = 0; i < _M; i++) {
        if (links[i] != node) {
          outdegree_table[node].push_back(links[i]);
        }
      }
    }
    return outdegree_table;
  }

  void computeEdgeLengthDistribution() {
    // #pragma omp parallel for default(none)                                         \
//     shared(_edge_length_distribution, _cur_num_nodes, _M, _hasher, _distance)
    for (node_id_t node = 0; node < _cur_num_nodes; node++) {
      node_id_t *links = getNodeLinks(node);
      for (size_t i = 0; i < _M; i++) {
        if (links[i] == node) {
          continue;
        }
        size_t hash = _hasher(node + links[i]);

        // #pragma omp critical {
        bool item_exists = _edge_length_distribution.find(hash) !=
                           _edge_length_distribution.end();
        if (item_exists) {
          continue;
        }
        // }

        // compute the distance between the two nodes
        float distance = _distance->distance(/* x = */ getNodeData(node),
                                             /* y = */ getNodeData(links[i]));

        // #pragma omp critical
        _edge_length_distribution[hash] = distance;
        // }
      }
    }
  }

  /**
   * @brief Store the new node in the global data structure. In a
   * multi-threaded setting, the index data guard should be held by the caller
   * with an exclusive lock.
   *
   * @param data The vector to add.
   * @param label The label (meta-data) of the vector.
   * @param new_node_id The id of the new node.
   */
  void allocateNode(void *data, label_t &label, node_id_t &new_node_id) {

    new_node_id = _cur_num_nodes;
    _distance->transformData(
        /* destination = */ getNodeData(new_node_id),
        /* src = */ data);
    *(getNodeLabel(new_node_id)) = label;

    node_id_t *links = getNodeLinks(new_node_id);
    // Initialize all edges to self
    std::fill_n(links, _M, new_node_id);
    _cur_num_nodes++;
  }

  /**
   * @brief Adds vectors to the index in batches.
   *
   * This method is responsible for adding vectors in batches, represented by
   * `data`, to the underlying graph. Each vector is associated with a label
   * provided in the `labels` vector. The method efficiently handles concurrent
   * additions by dividing the workload among multiple threads, defined by
   * `_num_threads`.
   *
   * The method ensures thread safety by employing locking mechanisms at the
   * node level in the underlying `connectNeighbors` and `beamSearch` methods.
   * This allows multiple threads to safely add vectors to the index without
   * causing data races or inconsistencies in the graph structure.
   *
   * @param data Pointer to the array of vectors to be added.
   * @param labels A vector of labels corresponding to each vector in `data`.
   * @param ef_construction Parameter for controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm. Must be greater than 0.
   *
   * @exception std::invalid_argument Thrown if `num_initializations` is less
   * than or equal to 0.
   * @exception std::runtime_error Thrown if the maximum number of nodes in the
   * index is reached.
   */
  void addBatch(void *data, std::vector<label_t> &labels, int ef_construction,
                int num_initializations = 100) {
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }
    uint32_t total_num_nodes = labels.size();
    uint32_t data_dimension = _distance->dimension();

    // Don't spawn any threads if we are only using one.
    if (_num_threads == 1) {
      for (uint32_t row_id = 0; row_id < total_num_nodes; row_id++) {
        void *vector = (float *)data + (row_id * data_dimension);
        label_t label = labels[row_id];
        this->add(vector, label, ef_construction, num_initializations);
      }
      return;
    }

    flatnav::executeInParallel(
        /* start_index = */ 0, /* end_index = */ total_num_nodes,
        /* num_threads = */ _num_threads, /* function = */
        [&](uint32_t row_index) {
          void *vector = (float *)data + (row_index * data_dimension);
          label_t label = labels[row_index];
          this->add(vector, label, ef_construction, num_initializations);
        });
  }

  /**
   * @brief Adds a single vector to the index.
   *
   * This method is called internally by `addBatch` for each vector in the
   * batch. The method ensures thread safety by using locking primitives,
   * allowing it to be safely used in a multi-threaded environment.
   *
   * The method first checks if the current number of nodes has reached the
   * maximum capacity. If so, it throws a runtime error. It then locks the index
   * structure to prevent concurrent modifications while allocating a new node.
   * After unlocking, it connects the new node to its neighbors in the graph.
   *
   * @param data Pointer to the vector data being added.
   * @param label Label associated with the vector.
   * @param ef_construction Parameter controlling the size of the dynamic
   * candidate list during the construction of the graph.
   * @param num_initializations Number of initializations for the search
   * algorithm.
   *
   * @exception std::runtime_error Thrown if the maximum number of nodes is
   * reached.
   */
  void add(void *data, label_t &label, int ef_construction,
           int num_initializations) {

    if (_cur_num_nodes >= _max_node_count) {
      throw std::runtime_error("Maximum number of nodes reached. Consider "
                               "increasing the `max_node_count` parameter to "
                               "create a larger index.");
    }
    _index_data_guard.lock();
    auto entry_node = initializeSearch(data, num_initializations);
    node_id_t new_node_id;
    allocateNode(data, label, new_node_id);
    _index_data_guard.unlock();

    if (new_node_id == 0) {
      return;
    }

    auto neighbors = beamSearch(
        /* query = */ data, /* entry_node = */ entry_node,
        /* buffer_size = */ ef_construction);

    selectNeighbors(/* neighbors = */ neighbors);
    connectNeighbors(neighbors, new_node_id);
  }

  /***
   * @brief Search the index for the k nearest neighbors of the query.
   * @param query The query vector.
   * @param K The number of nearest neighbors to return.
   * @param ef_search The search beam width.
   * @param num_initializations The number of random initializations to use.
   */
  std::vector<dist_label_t> search(const void *query, const int K,
                                   int ef_search,
                                   int num_initializations = 100) {
    node_id_t entry_node;
    if (_use_random_initialization) {
      entry_node = initializeSearchWithRandomness(query, num_initializations);
    } else {
      entry_node = initializeSearch(query, num_initializations);
    }
    PriorityQueue neighbors = beamSearch(/* query = */ query,
                                         /* entry_node = */ entry_node,
                                         /* buffer_size = */ std::max(K, ef_search));
    auto size = neighbors.size();
    std::vector<dist_label_t> results;
    results.reserve(size);
    while (!neighbors.empty()) {
      results.emplace_back(neighbors.top().first,
                           *getNodeLabel(neighbors.top().second));
      neighbors.pop();
    }
    std::sort(results.begin(), results.end(),
              [](const dist_label_t &left, const dist_label_t &right) {
                return left.first < right.first;
              });
    if (results.size() > static_cast<size_t>(K)) {
      results.resize(K);
    }

    return results;
  }

  void doGraphReordering(const std::vector<std::string> &reordering_methods) {

    for (const auto &method : reordering_methods) {
      auto outdegree_table = getGraphOutdegreeTable();
      std::vector<node_id_t> P;
      if (method == "gorder") {
        P = std::move(flatnav::gOrder<node_id_t>(outdegree_table, 5));
      } else if (method == "rcm") {
        P = std::move(flatnav::rcmOrder<node_id_t>(outdegree_table));
      } else {
        throw std::invalid_argument("Invalid reordering method: " + method);
      }

      relabel(P);
    }
  }

  void reorderGOrder(const int window_size = 5) {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P =
        flatnav::gOrder<node_id_t>(outdegree_table, window_size);

    relabel(P);
  }

  void reorderRCM() {
    auto outdegree_table = getGraphOutdegreeTable();
    std::vector<node_id_t> P = flatnav::rcmOrder<node_id_t>(outdegree_table);
    relabel(P);
  }

  static std::unique_ptr<Index<dist_t, label_t>>
  loadIndex(const std::string &filename) {
    std::ifstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for reading: " + filename);
    }

    cereal::BinaryInputArchive archive(stream);
    std::unique_ptr<Index<dist_t, label_t>> index(new Index<dist_t, label_t>());

    std::unique_ptr<DistanceInterface<dist_t>> dist =
        std::make_unique<dist_t>();

    // 1. Deserialize metadata
    archive(index->_M, index->_data_size_bytes, index->_node_size_bytes,
            index->_max_node_count, index->_cur_num_nodes, *dist);
    index->_visited_set_pool = new VisitedSetPool(
        /* initial_pool_size = */ 1,
        /* num_elements = */ index->_max_node_count);
    index->_distance = std::move(dist);
    index->_num_threads = std::thread::hardware_concurrency();
    index->_node_links_mutexes =
        std::vector<std::mutex>(index->_max_node_count);

    // 2. Allocate memory using deserialized metadata
    index->_index_memory =
        new char[index->_node_size_bytes * index->_max_node_count];

    // 3. Deserialize content into allocated memory
    archive(
        cereal::binary_data(index->_index_memory,
                            index->_node_size_bytes * index->_max_node_count));

    return index;
  }

  void saveIndex(const std::string &filename) {
    std::ofstream stream(filename, std::ios::binary);

    if (!stream.is_open()) {
      throw std::runtime_error("Unable to open file for writing: " + filename);
    }

    cereal::BinaryOutputArchive archive(stream);
    archive(*this);
  }

  inline void setNumThreads(uint32_t num_threads) {
    if (num_threads == 0 || num_threads > std::thread::hardware_concurrency()) {
      throw std::invalid_argument(
          "Number of threads must be greater than 0 and less than or equal to "
          "the number of hardware threads.");
    }
    _num_threads = num_threads;
    if (_num_threads == 1) {
      _visited_set_pool->setPoolSize(1);
    }
  }

  inline uint64_t getTotalIndexMemory() const {
    return static_cast<uint64_t>(_node_size_bytes * _max_node_count);
  }
  inline uint64_t mutexesAllocatedMemory() const {
    return static_cast<uint64_t>(_node_links_mutexes.size() *
                                 sizeof(std::mutex));
  }

  inline uint64_t visitedSetPoolAllocatedMemory() const {
    size_t pool_size = _visited_set_pool->poolSize();
    return static_cast<uint64_t>(pool_size * sizeof(VisitedSet));
  }

  inline uint32_t getNumThreads() const { return _num_threads; }

  inline size_t maxEdgesPerNode() const { return _M; }
  inline size_t dataSizeBytes() const { return _data_size_bytes; }

  inline size_t nodeSizeBytes() const { return _node_size_bytes; }

  inline size_t maxNodeCount() const { return _max_node_count; }

  inline size_t currentNumNodes() const { return _cur_num_nodes; }
  inline size_t dataDimension() const { return _distance->dimension(); }

  inline uint64_t distanceComputations() const {
    return _distance_computations.load();
  }

  void resetStats() {
    _distance_computations = 0;
    _metric_hops = 0;
  }
  
  // Return a reference to the node access counts
  inline const std::unordered_map<uint32_t, uint32_t> &
  getNodeAccessCounts() const {
    return _node_access_counts;
  }

  inline const std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t,
                                  pair_hash> &
  getEdgeAccessCounts() const {
    return _edge_access_counts;
  }

  inline const std::unordered_map<size_t, float> &
  getEdgeLengthDistribution() const {
    return _edge_length_distribution;
  }

  void getIndexSummary() const {
    std::cout << "\nIndex Parameters\n" << std::flush;
    std::cout << "-----------------------------\n" << std::flush;
    std::cout << "max_edges_per_node (M): " << _M << "\n" << std::flush;
    std::cout << "data_size_bytes: " << _data_size_bytes << "\n" << std::flush;
    std::cout << "node_size_bytes: " << _node_size_bytes << "\n" << std::flush;
    std::cout << "max_node_count: " << _max_node_count << "\n" << std::flush;
    std::cout << "cur_num_nodes: " << _cur_num_nodes << "\n" << std::flush;

    _distance->getSummary();
  }

private:
  friend class cereal::access;
  // Default constructor for cereal
  Index() = default;

  char *getNodeData(const node_id_t &n) const {
    return _index_memory + (n * _node_size_bytes);
  }

  node_id_t *getNodeLinks(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes;
    return reinterpret_cast<node_id_t *>(location);
  }

  label_t *getNodeLabel(const node_id_t &n) const {
    char *location = _index_memory + (n * _node_size_bytes) + _data_size_bytes +
                     (_M * sizeof(node_id_t));
    return reinterpret_cast<label_t *>(location);
  }

  inline void swapNodes(node_id_t a, node_id_t b, void *temp_data,
                        node_id_t *temp_links, label_t *temp_label) {

    // stash b in temp
    std::memcpy(temp_data, getNodeData(b), _data_size_bytes);
    std::memcpy(temp_links, getNodeLinks(b), _M * sizeof(node_id_t));
    std::memcpy(temp_label, getNodeLabel(b), sizeof(label_t));

    // place node at a in b
    std::memcpy(getNodeData(b), getNodeData(a), _data_size_bytes);
    std::memcpy(getNodeLinks(b), getNodeLinks(a), _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(b), getNodeLabel(a), sizeof(label_t));

    // put node b in a
    std::memcpy(getNodeData(a), temp_data, _data_size_bytes);
    std::memcpy(getNodeLinks(a), temp_links, _M * sizeof(node_id_t));
    std::memcpy(getNodeLabel(a), temp_label, sizeof(label_t));
  }

  /**
   * @brief Performs beam search for the nearest neighbors of the query.
   * @TODO: Add `entry_node_dist` argument to this function since we expect to
   * have computed that a priori.
   *
   * @param query               The query vector.
   * @param entry_node          The node to start the search from.
   * @param buffer_size         This is equivalent to `ef_search` in the HNSW
   *
   * @return PriorityQueue
   */
  PriorityQueue beamSearch(const void *query, const node_id_t entry_node,
                           const int buffer_size) {
    PriorityQueue neighbors;
    PriorityQueue candidates;

    auto *visited_set = _visited_set_pool->pollAvailableSet();
    visited_set->clear();

    float dist =
        _distance->distance(/* x = */ query, /* y = */ getNodeData(entry_node),
                            /* asymmetric = */ true);

    float max_dist = dist;
    candidates.emplace(-dist, entry_node);
    neighbors.emplace(dist, entry_node);
    visited_set->insert(entry_node);

    while (!candidates.empty()) {
      dist_node_t d_node = candidates.top();

      if ((-d_node.first) > max_dist && neighbors.size() >= buffer_size) {
        break;
      }
      candidates.pop();

      processCandidateNode(
          /* query = */ query, /* node = */ d_node.second,
          /* max_dist = */ max_dist, /* buffer_size = */ buffer_size,
          /* visited_set = */ visited_set,
          /* neighbors = */ neighbors, /* candidates = */ candidates);
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    return neighbors;
  }

  void processCandidateNode(const void *query, node_id_t &node, float &max_dist,
                            const int buffer_size, VisitedSet *visited_set,
                            PriorityQueue &neighbors,
                            PriorityQueue &candidates) {
    // Lock all operations on this specific node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[node]);
    float dist = 0.f;

    node_id_t *neighbor_node_links = getNodeLinks(node);
    for (uint32_t i = 0; i < _M; i++) {
      node_id_t neighbor_node_id = neighbor_node_links[i];

      // If using SSE, prefetch the next neighbor node data and the visited
      // marker
#ifdef USE_SSE
      if (i != _M - 1) {
        _mm_prefetch(getNodeData(neighbor_node_links[i + 1]), _MM_HINT_T0);
        visited_set->prefetch(neighbor_node_links[i + 1]);
      }
#endif

      bool neighbor_is_visited =
          visited_set->isVisited(/* num = */ neighbor_node_id);

      if (neighbor_is_visited) {
        continue;
      }
      visited_set->insert(/* num = */ neighbor_node_id);

      // Increment the counter in the visited map
      std::unique_lock<std::mutex> neighbor_lock(
          _node_links_mutexes[neighbor_node_id]);
      _node_access_counts[neighbor_node_id]++;
      // _node_access_counts_guard.lock();
      // // Increment the counter in the visited map
      // _node_access_counts[neighbor_node_id]++;
      // _node_access_counts_guard.unlock();

      _edge_access_counts_guard.lock();
      auto key = std::make_pair(node, neighbor_node_id);
      _edge_access_counts[key]++;
      _edge_access_counts_guard.unlock();

      neighbor_lock.unlock();

      dist = _distance->distance(/* x = */ query,
                                 /* y = */ getNodeData(neighbor_node_id),
                                 /* asymmetric = */ true);

      if (_collect_stats) {
        _distance_computations.fetch_add(1);
      }

      if (neighbors.size() < buffer_size || dist < max_dist) {
        candidates.emplace(-dist, neighbor_node_id);
        neighbors.emplace(dist, neighbor_node_id);
#ifdef USE_SSE
        _mm_prefetch(getNodeData(candidates.top().second), _MM_HINT_T0);
#endif
        if (neighbors.size() > buffer_size) {
          neighbors.pop();
        }
        if (!neighbors.empty()) {
          max_dist = neighbors.top().first;
        }
      }
    }
  }

  /**
   * @brief Selects neighbors from the PriorityQueue, according to the HNSW
   * heuristic. The neighbors priority queue contains elements sorted by
   * distance where the top element is the furthest neighbor from the query.
   */
  void selectNeighbors(PriorityQueue &neighbors) {
    if (neighbors.size() < _M) {
      return;
    }

    PriorityQueue candidates;
    std::vector<dist_node_t> saved_candidates;
    saved_candidates.reserve(_M);

    while (neighbors.size() > 0) {
      candidates.emplace(-neighbors.top().first, neighbors.top().second);
      neighbors.pop();
    }

    float cur_dist = 0.0;
    while (candidates.size() > 0) {
      if (saved_candidates.size() >= _M) {
        break;
      }
      // Extract the closest element from candidates.
      dist_node_t current_pair = candidates.top();
      candidates.pop();

      bool should_keep_candidate = true;
      for (const dist_node_t &second_pair : saved_candidates) {

        cur_dist =
            _distance->distance(/* x = */ getNodeData(second_pair.second),
                                /* y = */ getNodeData(current_pair.second));

        if (cur_dist < (-current_pair.first)) {
          should_keep_candidate = false;
          break;
        }
      }
      if (should_keep_candidate) {
        // We could do neighbors.emplace except we have to iterate
        // through saved_candidates, and std::priority_queue doesn't
        // support iteration (there is no technical reason why not).
        saved_candidates.push_back(current_pair);
      }
    }
    // TODO: implement my own priority queue, get rid of vector
    // saved_candidates, add directly to neighborqueue earlier.
    for (const dist_node_t &current_pair : saved_candidates) {
      neighbors.emplace(-current_pair.first, current_pair.second);
    }
  }

  void connectNeighbors(PriorityQueue &neighbors, node_id_t new_node_id) {
    // connects neighbors according to the HSNW heuristic

    // Lock all operations on this node
    std::unique_lock<std::mutex> lock(_node_links_mutexes[new_node_id]);

    node_id_t *new_node_links = getNodeLinks(new_node_id);
    int i = 0; // iterates through links for "new_node_id"

    while (neighbors.size() > 0) {
      node_id_t neighbor_node_id = neighbors.top().second;
      // add link to the current new node
      new_node_links[i] = neighbor_node_id;
      // now do the back-connections (a little tricky)

      std::unique_lock<std::mutex> neighbor_lock(
          _node_links_mutexes[neighbor_node_id]);
      node_id_t *neighbor_node_links = getNodeLinks(neighbor_node_id);
      bool is_inserted = false;
      for (size_t j = 0; j < _M; j++) {
        if (neighbor_node_links[j] == neighbor_node_id) {
          // If there is a self-loop, replace the self-loop with
          // the desired link.
          neighbor_node_links[j] = new_node_id;
          is_inserted = true;
          break;
        }
      }
      if (!is_inserted) {
        // now, we may to replace one of the links. This will disconnect
        // the old neighbor and create a directed edge, so we have to be
        // very careful. To ensure we respect the pruning heuristic, we
        // construct a candidate set including the old links AND our new
        // one, then prune this candidate set to get the new neighbors.

        float max_dist =
            _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                /* y = */ getNodeData(new_node_id));

        PriorityQueue candidates;
        candidates.emplace(max_dist, new_node_id);
        for (size_t j = 0; j < _M; j++) {
          if (neighbor_node_links[j] != neighbor_node_id) {
            auto label = neighbor_node_links[j];
            auto distance =
                _distance->distance(/* x = */ getNodeData(neighbor_node_id),
                                    /* y = */ getNodeData(label));
            candidates.emplace(distance, label);
          }
        }
        selectNeighbors(candidates);
        // connect the pruned set of candidates, including self-loops:
        size_t j = 0;
        while (candidates.size() > 0) { // candidates
          neighbor_node_links[j] = candidates.top().second;
          candidates.pop();
          j++;
        }
        while (j < _M) { // self-loops (unused links)
          neighbor_node_links[j] = neighbor_node_id;
          j++;
        }
      }

      // Unlock the current node we are iterating over
      neighbor_lock.unlock();

      // loop increments:
      i++;
      if (i >= _M) {
        i = _M;
      }
      neighbors.pop();
    }
  }

  /**
   * @brief Selects a node to use as the entry point for a new node.
   * This proceeds in a greedy fashion, by selecting the node with
   * the smallest distance to the query.
   *
   * @param query
   * @param num_initializations
   * @return node_id_t
   */
  node_id_t initializeSearch(const void *query, int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }

    int step_size = _cur_num_nodes / num_initializations;
    step_size = step_size ? step_size : 1;

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    if (_collect_stats) {
      _distance_computations.fetch_add(num_initializations);
    }

    for (node_id_t node = 0; node < _cur_num_nodes; node += step_size) {
      float dist =
          _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                              /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  // Use this during search to select a random entry point
  node_id_t initializeSearchWithRandomness(const void *query,
                                           int num_initializations) {
    // select entry_node from a set of random entry point options
    if (num_initializations <= 0) {
      throw std::invalid_argument(
          "num_initializations must be greater than 0.");
    }

    float min_dist = std::numeric_limits<float>::max();
    node_id_t entry_node = 0;

    for (int i = 0; i < num_initializations; i++) {
      node_id_t node = _distribution(_generator);
      float dist =
          _distance->distance(/* x = */ query, /* y = */ getNodeData(node),
                              /* asymmetric = */ true);
      if (dist < min_dist) {
        min_dist = dist;
        entry_node = node;
      }
    }
    return entry_node;
  }

  void relabel(const std::vector<node_id_t> &P) {
    // 1. Rewire all of the node connections
    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      node_id_t *links = getNodeLinks(n);
      for (int m = 0; m < _M; m++) {
        links[m] = P[links[m]];
      }
    }

    // 2. Physically re-layout the nodes (in place)
    char *temp_data = new char[_data_size_bytes];
    node_id_t *temp_links = new node_id_t[_M];
    label_t *temp_label = new label_t;

    auto *visited_set = _visited_set_pool->pollAvailableSet();

    // In this context, is_visited stores which nodes have been relocated
    // (it would be equivalent to name this variable "is_relocated").
    visited_set->clear();

    for (node_id_t n = 0; n < _cur_num_nodes; n++) {
      if (visited_set->isVisited(/* num = */ n)) {
        continue;
      }

      node_id_t src = n;
      node_id_t dest = P[src];

      // swap node at src with node at dest
      swapNodes(src, dest, temp_data, temp_links, temp_label);

      // mark src as having been relocated
      visited_set->insert(src);

      // recursively relocate the node from "dest"
      while (!visited_set->isVisited(/* num = */ dest)) {
        // mark node as having been relocated
        visited_set->insert(dest);
        // the value of src remains the same. However, dest needs
        // to change because the node located at src was previously
        // located at dest, and must be relocated to P[dest].
        dest = P[dest];

        // swap node at src with node at dest
        swapNodes(src, dest, temp_data, temp_links, temp_label);
      }
    }

    _visited_set_pool->pushVisitedSet(
        /* visited_set = */ visited_set);

    delete[] temp_data;
    delete[] temp_links;
    delete temp_label;
  }
}; // namespace flatnav

} // namespace flatnav