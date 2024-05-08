import json
import flatnav.index
import numpy as np
import argparse
import os
from typing import Tuple, Union
import logging
import hnswlib
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px

# import powerlaw
from utils import compute_metrics
from collections import Counter
from scipy.stats import skew

logging.basicConfig(level=logging.INFO)


# ROOT_DATASET_PATH = "/root/data/"
ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")

DATASET_NAMES = {
    "mnist-784-euclidean": "mnist",
    "sift-128-euclidean": "sift",
    "glove-100-angular": "glove",
    "cauchy-10-euclidean": "cauchy10",
    "cauchy-256-euclidean": "cauchy256",
    "cauchy-1024-euclidean": "cauchy1024",
    "normal-10-angular": "normal10-cosine",
    "normal-256-angular": "normal256-cosine",
    "normal-1024-angular": "normal1024-cosine",
    "normal-10-euclidean": "normal10-l2",
    "normal-256-euclidean": "normal256-l2",
    "normal-1024-euclidean": "normal1024-l2",
}


def depth_first_search(node: int, outdegree_table: np.ndarray, visited: set) -> None:
    visited.add(node)

    for neighbor_node in outdegree_table[node]:
        if neighbor_node not in visited:
            depth_first_search(
                node=neighbor_node, outdegree_table=outdegree_table, visited=visited
            )


def find_number_of_connected_components(
    outdegree_table: np.ndarray, subgraph: np.ndarray
) -> int:
    """
    Returns the number of connected components in the subgraph.
    :param outdegree_table: The outdegree table for the graph. This graph is assumed to be directed (can be cyclic).
    :param subgraph: The subgraph to compute the number of connected components for.
        This will just be a list of node ids. The corresponding edges will be in the outdegree table.
    """

    visited = set()
    num_connected_components = 0
    for node in subgraph:
        if node not in visited:
            depth_first_search(
                node=node, outdegree_table=outdegree_table, visited=visited
            )
            num_connected_components += 1

    return num_connected_components


# def plot_pmf_from_node_accesses(node_access_counts: dict, dataset_name: str) -> None:
#     """
#     Plots a PMF of the number of times nodes were accessed.
#     :param node_access_counts: A dictionary mapping node ids to the number of times they were accessed.
#     """
#     # Count how many times each access count occurs
#     access_count_frequencies = Counter(node_access_counts.values())
#     total_accesses = sum(access_count_frequencies.values())

#     # Sort the access counts and calculate their probabilities
#     sorted_access_counts = sorted(access_count_frequencies.keys())
#     probabilities = [access_count_frequencies[count] / total_accesses for count in sorted_access_counts]

#     # Plot the PMF
#     plt.figure(figsize=(10, 6))
#     plt.plot(sorted_access_counts, probabilities, 'ro')

#     powerlaw.plot_pdf(sorted_access_counts, color='black', linewidth=2)

#     plt.xticks(fontsize=20)
#     plt.yticks(fontsize=22)

#     plt.title(f"PMF of Node Access Counts in {dataset_name}")
#     plt.xlabel("Number of Times Accessed")
#     plt.ylabel("Probability")
#     plt.savefig(f"{dataset_name}_pmf-node-access-counts.png")


def plot_histogram(node_access_counts: dict, dataset_name: str) -> None:
    """
    Plots a histogram of the node access counts.
    :param node_access_counts: A dictionary mapping node ids to the number of times they were accessed.
    """

    x, y = [], []
    # We will estimate the emperical probability mass function by normalizing counts by
    # the total number of node accesses.
    total_node_accesses = sum(node_access_counts.values())
    for node_id, count in node_access_counts.items():
        x.append(node_id)
        y.append(count / total_node_accesses)

    # Plot the histogram
    plt.figure(figsize=(10, 10))
    plt.bar(x, y)
    plt.title(f"Dataset name: {dataset_name} --  Node access counts")
    plt.xlabel("Node id")
    plt.ylabel("$P(N)$")
    plt.savefig(f"{dataset_name}node-access-counts.png")

    # Compute the skewness of the node access counts
    # node_access_counts = np.array(list(node_access_counts.values()))
    # skewness = pd.Series(node_access_counts).skew()
    # logging.info(f"Skewness of node access counts: {skewness}")

    # # Plot the histogram
    # plt.figure(figsize=(10, 10))
    # sns.histplot(node_access_counts, bins=90, kde=True, log_scale=(True, True))
    # plt.title(
    #     f"Dataset name: {dataset_name} --  Node access counts (skewness: {skewness:.3f})"
    # )
    # plt.xscale("log")
    # plt.yscale("log")

    # plt.xlabel("Node access counts, N")
    # plt.ylabel("$P(N)$")

    # plt.savefig(f"{dataset_name}node-access-counts.png")

    logging.info(f"Plot saved at {dataset_name}node-access-counts.png")


def get_node_access_counts_distribution(
    dataset_name: str,
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k: int,
):
    """
    Computes the node access counts distribution for a given index.
    :param index: The index to compute the node access counts distribution for.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param ef_search: The ef-search parameter.
    :param k: The number of nearest neighbors to find.
    """

    index.set_num_threads(1)
    requested_metrics = [f"recall@{k}", "latency", "qps"]
    flatnav_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
    )
    logging.info(f"FlatNav metrics: {flatnav_metrics}")

    node_access_counts: dict = index.get_node_access_counts()
    return node_access_counts

    # name = DATASET_NAMES[dataset_name]
    # plot_pmf_from_node_accesses(node_access_counts=node_access_counts, dataset_name=name)
    # plot_histogram(node_access_counts=node_access_counts, dataset_name=name)
    # fit_power_law(distribution=node_access_counts, save_path=f"{name}-powerlaw.png")


def plot_kde_distributions(distributions, bw_adjust_value=0.3):
    plt.figure(figsize=(10, 6), dpi=300)
    ax = plt.gca()  # Get the current Axes instance

    # Compute skewness before the log transformation and store them
    skewness_values = {}
    for dataset_name, node_access_counts in distributions.items():
        skewness_values[dataset_name] = skew(list(node_access_counts.values()))

    for dataset_name, node_access_counts in distributions.items():
        # Apply log-transform to the counts, adding 1 to avoid log(0)
        log_counts = np.log1p(list(node_access_counts.values()))
        raw_skewness = skewness_values[dataset_name]
        # Replace "euclidean" with "l2" and "angular" with "cosine"
        dataset_name = dataset_name.replace("euclidean", "l2").replace(
            "angular", "cosine"
        )

        # Plot the KDE for log-transformed data with less smoothness
        sns.kdeplot(
            log_counts,
            label=f"{dataset_name} ($\\tilde{{\\mu}}_3$ = {raw_skewness:.2f})",
            bw_adjust=bw_adjust_value,
        )

    # Set up the legend on the right of the plot
    plt.legend(title="Dataset", loc="center left", bbox_to_anchor=(1, 0.5))

    # Improve plot aesthetics
    sns.despine(trim=True)  # Trim the spines for a cleaner look
    plt.grid(True)  # Add gridlines
    plt.xlabel("Log of Node access counts")
    plt.ylabel("PDF")
    plt.title("KDE of Log-Transformed Node Access Counts")

    # Adjust the plot area to fit the legend and increase the resolution
    plt.subplots_adjust(right=0.75)
    plt.tight_layout()

    plt.savefig("node_access_distributions.png")


def plot_distributions(distributions):
    fig, ax = plt.subplots()

    # Generate bins for the histograms
    all_counts = [count for dist in distributions.values() for count in dist.values()]
    max_count = max(all_counts)
    bins = (
        np.arange(0, max_count + 2) - 0.5
    )  # 0.5 offset to center the bars on the integers

    for dataset_name, node_access_counts in distributions.items():
        # Convert counts to frequencies
        total_visits = sum(node_access_counts.values())
        frequencies = np.array(list(node_access_counts.values())) / total_visits

        # Plot the PDF or PMF
        hist, _ = np.histogram(
            list(node_access_counts.values()), bins=bins, density=True
        )
        mid_points = 0.5 * (bins[1:] + bins[:-1])
        plt.plot(mid_points, hist, label=dataset_name)

    plt.xlabel("In-degree (bin number)")
    plt.ylabel("PDF")
    plt.legend(title="Dataset")

    plt.savefig("node_access_distributions.png")


def get_edge_lengths_distribution(
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex], dataset_name: str
):
    # This is a dictionary mapping unique edge hashes to their lengths.
    # The lengths are floating point numbers.
    # We want to plot a histogram of this distribution.
    distribution: dict = index.get_edge_length_distribution()

    # Normalize the distribution by the largest edge length
    # so that the histogram is easier to interpret
    max_edge_length = max(distribution.values())
    distribution = {k: v / max_edge_length for k, v in distribution.items()}

    skewness = pd.Series(distribution.values()).skew()

    # Plot the histogram
    plt.figure(figsize=(10, 10))
    sns.histplot(distribution.values(), bins="auto", kde=False, log_scale=(False, True))
    plt.title(f"{dataset_name} Edge length distribution (skewness: {skewness:.3f})")
    plt.xlabel("Edge length")
    plt.ylabel("Number of edges")
    plt.savefig(f"{dataset_name}_edge_lengths.png")

    logging.info(f"Plot saved at {dataset_name}_edge_lengths.png")


def main(
    dataset_name: str,
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    distribution_type: str,
    num_initializations: int = 100,
) -> Tuple[dict, dict]:
    """
    Computes the following metrics for FlatNav and HNSW:
        - Recall@k
        - Latency
        - QPS
        - Hubness score as measured by the skewness of the k-occurence distribution (N_k)

    NOTE: Index construction is done in parallel, but search is single-threaded.

    :param dataset_name: The name of the dataset.
    :param train_dataset: The dataset to compute the skewness for.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param distance_type: The distance type to use for computing the skewness.
    :param max_edges_per_node: The maximum number of edges per node.
    :param ef_construction: The ef-construction parameter.
    :param ef_search: The ef-search parameter.
    :param k: The number of nearest neighbors to find.
    :param distribution_type: The desired distribution to consider. Can be either 'node-access' or 'edge-length'.
    :param num_initializations: The number of initializations for FlatNav.
    """

    dataset_size, dim = train_dataset.shape
    # hnsw_index = hnswlib.Index(
    #     space=distance_type if distance_type == "l2" else "ip", dim=dim
    # )
    # hnsw_index.init_index(
    #     max_elements=dataset_size,
    #     ef_construction=ef_construction,
    #     M=max_edges_per_node // 2,
    # )

    # hnsw_index.set_num_threads(os.cpu_count())

    # logging.debug(f"Building index...")
    # hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    # hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    # hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        use_random_initialization=True,
        random_seed=42,
    )

    flatnav_index.set_num_threads(os.cpu_count())

    # Train the index.
    start = time.time()
    flatnav_index.add(
        data=train_dataset, ef_construction=ef_construction, num_initializations=300
    )
    end = time.time()

    # Here we will first allocate memory for the index and then build edge connectivity
    # using the HNSW base layer graph. We do not use the ef-construction parameter since
    # it's assumed to have been used when building the HNSW base layer.
    # flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()

    # Now delete the HNSW base layer graph since we don't need it anymore
    # os.remove(hnsw_base_layer_filename)

    if distribution_type.lower() == "node-access":
        node_access_counts = get_node_access_counts_distribution(
            dataset_name=dataset_name,
            index=flatnav_index,
            queries=queries,
            ground_truth=ground_truth,
            ef_search=ef_search,
            k=k,
        )

        return node_access_counts

    elif distribution_type.lower() == "edge-length":
        name = DATASET_NAMES[dataset_name]
        get_edge_lengths_distribution(index=flatnav_index, dataset_name=name)
    else:
        raise ValueError(f"Invalid distribution type: {distribution_type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="dataset names. All will be expected to be at the same path.",
    )

    parser.add_argument(
        "--k",
        type=int,
        required=False,
        default=100,
        help="Number of nearest neighbors to consider",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="Distance/metric to use (l2 or angular)",
    )
    parser.add_argument(
        "--ef-construction", type=int, required=True, help="ef-construction parameter."
    )

    parser.add_argument(
        "--ef-search", type=int, required=True, help="ef-search parameter."
    )

    parser.add_argument(
        "--num-node-links",
        type=int,
        required=True,
        help="max-edges-per-node parameter.",
    )

    parser.add_argument(
        "--distribution-type",
        type=str,
        required=True,
        help="Desired distribution to consider. Can be either 'node-access' or 'edge-length'.",
    )

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy").astype(np.float32, copy=False),
        np.load(f"{base_path}/{dataset_name}.test.npy").astype(np.float32, copy=False),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy").astype(np.uint32, copy=False),
    )


if __name__ == "__main__":
    args = parse_args()

    dataset_names = args.datasets
    distance_types = args.metrics
    if len(dataset_names) != len(distance_types):
        raise RuntimeError("Number of datasets and metrics/distances must be the same")

    # Map from dataset name to node access counts
    distributions = {}

    for index, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset {dataset_name}...")
        # metric = distance_types[index]
        # base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        # if not os.path.exists(base_path):
        #     # Create the directory if it doesn't exist
        #     raise ValueError(f"Dataset path not found at {base_path}")

        # train_dataset, queries, ground_truth = load_dataset(
        #     base_path=base_path, dataset_name=dataset_name
        # )

        # node_access_counts = main(
        #     dataset_name=dataset_name,
        #     train_dataset=train_dataset,
        #     queries=queries,
        #     ground_truth=ground_truth,
        #     distance_type=metric,
        #     max_edges_per_node=args.num_node_links,
        #     ef_construction=args.ef_construction,
        #     ef_search=args.ef_search,
        #     k=args.k,
        #     distribution_type=args.distribution_type,
        # )

        # Save the node access counts for this dataset to a JSON file
        # with open(f"{dataset_name}_node_access_counts.json", "w") as f:
        #     json.dump(node_access_counts, f)

        # Load the node access counts from the JSON file
        with open(f"{dataset_name}_node_access_counts.json", "r") as f:
            node_access_counts = json.load(f)

        distributions[dataset_name] = node_access_counts

    # Now plot the distributions
    # plot_distributions(distributions)
    plot_kde_distributions(distributions)
