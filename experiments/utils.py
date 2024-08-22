from typing import Tuple, Union, List, Optional
import numpy as np
import hnswlib
import time
import flatnav
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


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


def get_metric_from_dataset_name(dataset_name: str) -> str:
    """
    Extract the metric from the dataset name. The metric is the last part of the dataset name.
    Ex. normal-10-euclidean -> l2
        mnist-784-euclidean -> l2
        normal-10-angular -> angular
    """
    metric = dataset_name.split("-")[-1]
    if metric == "euclidean":
        return "l2"
    elif metric == "angular":
        return "angular"
    raise ValueError(f"Invalid metric: {metric}")

def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy"),
        np.load(f"{base_path}/{dataset_name}.test.npy"),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy"),
    )

def load_graph_from_mtx_file(mtx_filename: str) -> list[list[int]]:
    """
    Loads a graph from a .mtx file and returns it as an adjacency list.
    :param mtx_filename: Path to the .mtx file.
    :return: Adjacency list representation of the graph.
    """
    with open(mtx_filename, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if not line.startswith("%")]

    num_nodes, num_edges = map(int, lines[0].split())
    graph = [[] for _ in range(num_nodes)]

    for line in lines[1:]:
        u, v = map(int, line.split())
        # Adjust for 1-based indexing 
        u -= 1
        v -= 1
        graph[u].append(v)
        graph[v].append(u)

    return graph