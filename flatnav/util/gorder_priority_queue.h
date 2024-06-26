#pragma once

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace flatnav {

template <typename node_id_t> class GorderPriorityQueue {

  typedef std::unordered_map<node_id_t, int> map_t;

  struct Node {
    node_id_t key;
    int priority;
  };

  std::vector<Node> _list;
  map_t _index_table; // map: key -> index in _list

  inline void swap(int i, int j) {
    Node tmp = _list[i];
    _list[i] = _list[j];
    _list[j] = tmp;
    _index_table[_list[i].key] = i;
    _index_table[_list[j].key] = j;
  }

public:
  GorderPriorityQueue(const std::vector<node_id_t> &nodes) {
    for (int i = 0; i < nodes.size(); i++) {
      _list.push_back({nodes[i], 0});
      _index_table[nodes[i]] = i;
    }
  }

  GorderPriorityQueue(size_t N) {
    for (unsigned int i = 0; i < N; i++) {
      _list.push_back({i, 0});
      _index_table[i] = i;
    }
  }

  void print() {
    for (int i = 0; i < _list.size(); i++) {
      std::cout << "(" << _list[i].key << ":" << _list[i].priority << ")"
                << " ";
    }
    std::cout << std::endl;
  }

  static bool compare(const Node &a, const Node &b) {
    return (a.priority < b.priority);
  }

  void increment(node_id_t key) {
    typename map_t::const_iterator i = _index_table.find(key);
    if (i == _index_table.end()) {
      return;
    }
    // int new_index = _list.size()-1;
    // while((new_index > 0) && (_list[new_index].priority >
    // _list[i->second].priority)){ 	new_index--;
    // }

    auto it =
        std::upper_bound(_list.begin(), _list.end(), _list[i->second], compare);
    size_t new_index = it - _list.begin() - 1; // possible bug
    // new_index points to the right-most element with same priority as key
    // i.e. priority equal to "_list[i->second].priority" (i.e. the current
    // priority)
    swap(i->second, new_index);
    _list[new_index].priority++;
  }

  void decrement(node_id_t key) {
    typename map_t::const_iterator i = _index_table.find(key);
    if (i == _index_table.end()) {
      return;
    }
    // int new_index = _list.size()-1;
    // while((new_index > 0) && (_list[new_index].priority >=
    // _list[i->second].priority)){ 	new_index--;
    // }
    // new_index++;
    // i shoudl do this better but am pressed for time now
    auto it =
        std::lower_bound(_list.begin(), _list.end(), _list[i->second], compare);
    size_t new_index = it - _list.begin(); // POSSIBLE BUG
    // while((new_index > _list.size()) && (_list[new_index].priority ==
    // _list[i->second].priority)){ 	new_index++;
    // }
    // new_index--;
    // new_index points to the right-most element with same priority as key

    swap(i->second, new_index);
    _list[new_index].priority--;
  }

  node_id_t pop() {
    Node max = _list.back();
    _list.pop_back();
    _index_table.erase(max.key);
    return max.key;
  }

  size_t size() { return _list.size(); }
};

} // namespace flatnav