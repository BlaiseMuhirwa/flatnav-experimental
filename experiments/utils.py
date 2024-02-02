from typing import Tuple, Union, List, Optional
import numpy as np
import hnswlib
import time
import flatnav


def search_in_batches(index, queries, batch_size, ef_search, k):
    if type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex):
        for i in range(0, len(queries), batch_size):
            top_k_indices_batch, _ = index.search(
                queries=queries[i : i + batch_size],
                ef_search=ef_search,
                K=k,
                num_initializations=300,
            )
            yield top_k_indices_batch
    else:
        index.set_ef(ef_search)
        for i in range(0, len(queries), batch_size):
            top_k_indices_batch, _ = index.knn_query(
                data=queries[i : i + batch_size], k=k
            )
            yield top_k_indices_batch


# def search_in_batches(
#     index, queries, batch_size, ef_search, k
# ) -> Tuple[np.ndarray, float]:
#     top_k_indices = []
#     if type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex):
#         start = time.time()
#         for i in range(0, len(queries), batch_size):
#             top_k_indices_batch, _ = index.search(
#                 queries=queries[i : i + batch_size],
#                 ef_search=ef_search,
#                 K=k,
#                 num_initializations=300,
#             )
#             top_k_indices.append(top_k_indices_batch)
#         end = time.time()
#     else:
#         index.set_ef(ef_search)
#         start = time.time()
#         for i in range(0, len(queries), batch_size):
#             top_k_indices_batch, _ = index.knn_query(
#                 data=queries[i : i + batch_size], k=k
#             )
#             top_k_indices.append(top_k_indices_batch)
#         end = time.time()

#     querying_time = end - start
#     top_k_indices = np.concatenate(top_k_indices)
#     return top_k_indices, querying_time


def compute_metrics(
    requested_metrics: List[str],
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
    batch_size: Optional[int] = None,
) -> dict:
    """
    Compute recall and QPS for given queries, ground truth for the given index(FlatNav or HNSW).

    Args:
        - requested_metrics: A list of metrics to compute. These include recall, qps and latency.
        - index: A FlatNav index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.
        - ef_search: The number of neighbors to visit during search.
        - batch_size: The number of queries to search in a batch. If None, search all queries at once.

    Returns:
        - A dictionary containing the requested metrics.

    """

    index_is_flatnav = type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex)

    # Use teh batch size to search in batches and then concatenate the results
    if batch_size is not None:
        start = time.time()
        top_k_indices = []
        for batch_result in search_in_batches(
            index=index,
            queries=queries,
            batch_size=batch_size,
            ef_search=ef_search,
            k=k,
        ):
            top_k_indices.append(batch_result)

        top_k_indices = np.concatenate(top_k_indices)
        end = time.time()
        querying_time = end - start

    else:
        if type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex):
            start = time.time()
            top_k_indices, _ = index.search(
                queries=queries, ef_search=ef_search, K=k, num_initializations=300
            )
            end = time.time()
            querying_time = end - start
        else:
            index.set_ef(ef_search)
            start = time.time()
            top_k_indices, _ = index.knn_query(data=queries, k=k)
            end = time.time()
            querying_time = end - start

    metrics = {}

    if "qps" in requested_metrics:
        qps = len(queries) / querying_time
        metrics["qps"] = qps

    if "latency" in requested_metrics:
        # Get latency in milliseconds
        latency = querying_time * 1000 / len(queries)
        metrics["latency"] = latency

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)

    metrics["recall"] = recall

    return metrics
