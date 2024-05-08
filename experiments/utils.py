from typing import Tuple, Union, List, Optional
import numpy as np
import hnswlib
import time
import flatnav
import os
from sklearn.neighbors import NearestNeighbors
import powerlaw
import matplotlib.pyplot as plt


def fit_power_law(distribution: dict, save_path: Optional[str] = None) -> None:
    data = np.array(list(distribution.values()))

    # Fit the power-law distribution
    fit = powerlaw.Fit(data, discrete=True)
    xmin = fit.xmin
    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma

    print(f"Xmin: {xmin}, alpha: {alpha}, sigma: {sigma}")

    R, p = fit.distribution_compare("power_law", "lognormal")
    print(f"Likelihood Ratio R: {R}, p-value: {p}")

    # Plot the PDF
    fig = fit.plot_pdf(color="b", linewidth=2)
    fit.power_law.plot_pdf(color="b", linestyle="--", ax=fig)

    # Add alpha and sigma to the plot
    plt.text(
        0.5, 0.5, f"alpha: {alpha:.3f}\nsigma: {sigma:.3f}", transform=fig.transAxes
    )

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path)


def generate_iid_normal_dataset(
    num_samples: int,
    num_dimensions: int,
    num_queries: int,
    k: int,
    directory_path: str,
    dataset_name: str,
    metric: str = "cosine",
):
    """
    Generatest a dataset with the specified number of samples and dimensions using
    the standard normal distribution.
    Separates a subset for queries and computes their true k nearest neighbors.
    :param num_samples: Number of samples in the dataset.
    :param num_dimensions: Number of dimensions for each sample.
    :param num_queries: Number of queries to be separated from the dataset.
    :param k: The number of nearest neighbors to find.
    :param directory_path: Base path to save the dataset, queries, and ground truth labels.
    :param dataset_name: Name of the dataset (should be something like normal-10-angular)
    :param metric: Metric to use for computing nearest neighbors.
    """

    dataset = np.random.normal(size=(num_samples, num_dimensions))
    np.random.shuffle(dataset)
    query_set = dataset[:num_queries]
    dataset_without_queries = dataset[num_queries:]
    neighbors = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric).fit(
        dataset_without_queries
    )
    ground_truth_labels = neighbors.kneighbors(query_set, return_distance=False)

    # Normalize the dataset and queries if using cosine distance
    if metric in ["cosine", "angular", "ip"]:
        dataset_without_queries /= (
            np.linalg.norm(dataset_without_queries, axis=1, keepdims=True) + 1e-30
        )
        query_set /= np.linalg.norm(query_set, axis=1, keepdims=True) + 1e-30

    # Create directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Save the dataset
    np.save(
        f"{directory_path}/{dataset_name}.train.npy",
        dataset_without_queries.astype(np.float32),
    )
    np.save(f"{directory_path}/{dataset_name}.test.npy", query_set.astype(np.float32))
    np.save(
        f"{directory_path}/{dataset_name}.gtruth.npy",
        ground_truth_labels.astype(np.int32),
    )


def generate_cauchy_dataset(
    num_samples: int,
    num_dimensions: int,
    num_queries: int,
    k: int,
    base_path: str,
    metric: str = "minkowski",
    p: int = 2,
):
    """
    Generates a dataset with the specified number of samples and dimensions using
    the Cauchy distribution.
    Separates a subset for queries and computes their true k nearest neighbors.

    :param num_samples: Number of samples in the dataset.
    :param num_dimensions: Number of dimensions for each sample.
    :param num_queries: Number of queries to be separated from the dataset.
    :param k: The number of nearest neighbors to find.
    :param base_path: Base path to save the dataset, queries, and ground truth labels.
    :param metric: Metric to use for computing nearest neighbors.
    :param p: Parameter for the metric.

    NOTE: metric="minkowski" and p=2 is equivalent to Euclidean distance.
    See: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
    """
    # Generate the dataset
    dataset = np.random.standard_cauchy(size=(num_samples, num_dimensions))

    # Separate out a subset for queries
    np.random.shuffle(dataset)
    query_set = dataset[:num_queries]
    dataset_without_queries = dataset[num_queries:]

    # Compute the true k nearest neighbors for the query set
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="brute", p=p, metric=metric).fit(
        dataset_without_queries
    )
    ground_truth_labels = nbrs.kneighbors(query_set, return_distance=False)

    # Save the dataset without queries, the queries, and the ground truth labels
    np.save(f"{base_path}/train.npy", dataset_without_queries.astype(np.float32))
    np.save(f"{base_path}/test.npy", query_set.astype(np.float32))
    np.save(f"{base_path}/ground_truth.npy", ground_truth_labels.astype(np.int32))


def compute_k_occurence_distrubution(top_k_indices: np.ndarray) -> np.ndarray:
    """
    Computes the distribution of k-occurences for each node in the given array.
    :param top_k_indices: array of shape (dataset_size, k) containing the indices of
            the k nearest neighbors for each node.

    :return: array of shape (dataset_size,) containing the k-occurence distribution for each node (N_k)
    """

    # validate indices. If any value is negative, throw an error
    if np.any(top_k_indices < 0):
        raise ValueError("Indices cannot be negative")

    dataset_size = top_k_indices.shape[0]
    k_occurence_distribution = np.zeros(dataset_size, dtype=int)

    flattened_indices = top_k_indices.flatten()
    unique_indices, counts = np.unique(flattened_indices, return_counts=True)
    k_occurence_distribution[unique_indices] = counts

    return k_occurence_distribution


def compute_skewness(
    index: Union[flatnav.index.L2Index, hnswlib.Index],
    dataset: np.ndarray,
    ef_search: int,
    k: int,
) -> float:
    is_flatnav_index = type(index) in [flatnav.index.L2Index, flatnav.index.IPIndex]
    if is_flatnav_index:
        _, top_k_indices = index.search(
            queries=dataset,
            ef_search=ef_search,
            K=k,
        )
    elif type(index) == hnswlib.Index:
        top_k_indices, _ = index.knn_query(dataset, k=k)
    else:
        raise ValueError("Invalid index")

    k_occurence_distribution = compute_k_occurence_distrubution(
        top_k_indices=top_k_indices
    )
    mean = np.mean(k_occurence_distribution)
    std_dev = np.std(k_occurence_distribution)
    denominator = len(k_occurence_distribution) * (std_dev**3)
    skewness = (np.sum((k_occurence_distribution - mean) ** 3)) / denominator

    return skewness


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
) -> dict[str, float]:
    """
    Compute recall and QPS for given queries, ground truth for the given index(FlatNav or HNSW).

    Args:
        - requested_metrics: A dict containing the requested metrics.
                Supported metrics include: recall, QPS and average query latency.
        - index: A FlatNav index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.
        - ef_search: The number of neighbors to visit during search.
        - batch_size: The number of queries to search in a batch. If None, search all queries at once.

    Returns:
        A dict containing the requested metrics.
    """
    metrics = {}

    if type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex):
        print(f"[FlatNav] searching with num-threads = {index.num_threads}")
        start = time.time()
        _, top_k_indices = index.search(
            queries=queries, ef_search=ef_search, K=k, num_initializations=100
        )
        end = time.time()
    else:
        print(f"[HNSW] search")
        index.set_ef(ef_search)
        start = time.time()
        # Search for HNSW return (ids, distances) instead of (distances, ids)
        top_k_indices, _ = index.knn_query(data=queries, k=k)
        end = time.time()

    querying_time = end - start
    if "latency" in requested_metrics:
        latency = querying_time / len(queries)
        latency *= 1000
        # Add latency in milliseconds
        metrics["latency"] = latency

    if "qps" in requested_metrics:
        metrics["qps"] = len(queries) / querying_time

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)

    metrics[f"recall@{k}"] = recall

    return metrics
