#pragma once

// #include <flatnav/util/SIMDDistanceSpecializations.h>

#include <cstring>
#include <flatnav/util/macros.h>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdint.h>
#include <thread>
#include <vector>

namespace flatnav {

class VisitedSet {
private:
  uint8_t _mark;
  uint8_t *_table;
  uint32_t _table_size;

public:
  VisitedSet(const uint32_t size) : _mark(1), _table_size(size) {
    // initialize values to 0
    _table = new uint8_t[_table_size]();
  }

  inline void prefetch(const uint32_t num) const {
#ifdef USE_SSE
    _mm_prefetch(reinterpret_cast<const char *>(&_table[num]), _MM_HINT_T0);
#endif
  }

  inline uint8_t getMark() const { return _mark; }

  inline void insert(const uint32_t num) { _table[num] = _mark; }

  inline uint32_t size() const { return _table_size; }

  inline void clear() {
    _mark++;
    if (_mark == 0) {
      std::memset(_table, 0, _table_size);
      _mark = 1;
    }
  }

  inline bool isVisited(const uint32_t num) const {
    return _table[num] == _mark;
  }

  ~VisitedSet() { delete[] _table; }

  // copy constructor
  VisitedSet(const VisitedSet &other)
      : _table_size(other._table_size), _mark(other._mark) {

    _table = new uint8_t[_table_size];
    std::memcpy(_table, other._table, _table_size);
  }

  // move constructor
  VisitedSet(VisitedSet &&other) noexcept
      : _table_size(other._table_size), _mark(other._mark),
        _table(other._table) {
    other._table = nullptr;
    other._table_size = 0;
    other._mark = 0;
  }

  // copy assignment
  VisitedSet &operator=(const VisitedSet &other) {
    if (this != &other) {
      delete[] _table;
      _table_size = other._table_size;
      _mark = other._mark;
      _table = new uint8_t[_table_size];
      std::memcpy(_table, other._table, _table_size);
    }
    return *this;
  }

  // move assignment
  VisitedSet &operator=(VisitedSet &&other) noexcept {
    _table_size = other._table_size;
    _mark = other._mark;
    _table = other._table;
    other._table = NULL;
    other._table_size = 0;
    other._mark = 0;
    return *this;
  }
};

/**
 *
 * @brief Manages a pool of VisitedSet objects in a thread-safe manner.
 *
 * This class is designed to efficiently provide and manage a pool of
 * VisitedSet instances for concurrent use in multi-threaded
 * environments. It ensures that each visited set can be used by only one thread
 * at a time without the risk of concurrent access and modification.
 *
 * The class preallocates a specified number of VisitedSet objects to
 * eliminate the overhead of dynamic allocation during runtime. It uses a mutex
 * to synchronize access to the visisted set pool, ensuring that only one thread
 * can modify the pool at any given time. This mechanism provides both thread
 * safety and improved performance by reusing visited_set objects instead of
 * continuously creating and destroying them.
 *
 * When a thread requires a VisitedSet, it can call
 * `pollAvailablevisited_set()` to retrieve an available visited_set from the
 * pool. If the pool is empty, the function will dynamically allocate a new
 * visited_set to ensure that the requesting thread can proceed with its task.
 * Once the thread has finished using the visited_set, it should return it to
 * the pool by calling `pushvisited_set()`.
 *
 * @note The class assumes that all threads will properly return the
 * visited_sets to the pool after use. Failing to return a visited_set will
 * deplete the pool and lead to dynamic allocation, negating the performance
 * benefits.
 *
 * Usage example:
 * @code
 * VisitedSetPool visited_pool(10, 1000);
 * VisitedSet* visited_set = visited_set_pool.pollAvailableSet();
 * // Use the visited_set in a thread...
 * visited_set_pool.pushVisitedSet(visited_set);
 * @endcode
 *
 * @param initial_pool_size The number of visited_set objects to initially
 * create and store in the pool.
 * @param num_elements The size of each VisitedSet, which typically
 * corresponds to the number of nodes or elements that each visited_set is
 * expected to manage.
 */
class VisitedSetPool {
  std::vector<VisitedSet *> _visisted_set_pool;
  std::mutex _pool_guard;
  uint32_t _num_elements;
  uint32_t _max_pool_size;

public:
  VisitedSetPool(uint32_t initial_pool_size, uint32_t num_elements,
                 uint32_t max_pool_size = std::thread::hardware_concurrency())
      : _visisted_set_pool(initial_pool_size), _num_elements(num_elements),
        _max_pool_size(max_pool_size) {
    if (initial_pool_size > max_pool_size) {
      throw std::invalid_argument(
          "initial_pool_size must be less than or equal to max_pool_size");
    }
    for (uint32_t visited_set_id = 0;
         visited_set_id < _visisted_set_pool.size(); visited_set_id++) {
      _visisted_set_pool[visited_set_id] =
          new VisitedSet(/* size = */ _num_elements);
    }
  }

  // TODO: Enforce the condition that we never allocate more than _max_pool_size
  // visited_sets. For now there is nothing stopping a user from allocating more
  // than _max_pool_size.
  VisitedSet *pollAvailableSet() {
    std::unique_lock<std::mutex> lock(_pool_guard);

    if (!_visisted_set_pool.empty()) {
      auto *visited_set = _visisted_set_pool.back();
      _visisted_set_pool.pop_back();
      return visited_set;
    } else {
      return new VisitedSet(/* size = */ _num_elements);
    }
  }

  size_t poolSize() const { return _visisted_set_pool.size(); }

  void pushVisitedSet(VisitedSet *visited_set) {
    std::unique_lock<std::mutex> lock(_pool_guard);

    _visisted_set_pool.push_back(visited_set);
  }

  void setPoolSize(uint32_t new_pool_size) {
    std::unique_lock<std::mutex> lock(_pool_guard);

    if (new_pool_size > _visisted_set_pool.size()) {
      throw std::invalid_argument(
          "new_pool_size must be less than or equal to the current pool size");
    }

    while (_visisted_set_pool.size() > new_pool_size) {
      auto *visited_set = _visisted_set_pool.back();
      _visisted_set_pool.pop_back();
      delete visited_set;
    }
  }

  inline uint32_t getPoolSize() { return _visisted_set_pool.size(); }

  ~VisitedSetPool() {
    while (!_visisted_set_pool.empty()) {
      auto *visited_set = _visisted_set_pool.back();
      _visisted_set_pool.pop_back();
      delete visited_set;
    }
  }
};

} // namespace flatnav