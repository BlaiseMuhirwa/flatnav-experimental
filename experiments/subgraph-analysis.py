import json
import numpy as np
import os
import argparse
from typing import Tuple, List, Dict
from utils import compute_metrics
import flatnav
import time
import logging
import hnswlib
import pickle

logging.basicConfig(level=logging.INFO)
# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")
ROOT_DATASET_PATH = "/root/data"


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
) -> List[List[int]]:
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
    hnsw_index = hnswlib.Index(
        space=distance_type if distance_type == "l2" else "ip", dim=dim
    )
    hnsw_index.init_index(
        max_elements=dataset_size,
        ef_construction=ef_construction,
        M=max_edges_per_node // 2,
    )

    hnsw_index.set_num_threads(os.cpu_count())

    logging.debug(f"Building index...")
    hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        mtx_filename=hnsw_base_layer_filename,
        use_random_initialization=True,
        random_seed=42,
    )

    flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()

    # outdegree_table = flatnav_index.get_graph_outdegree_table()
    # return outdegree_table

    # flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()
    os.remove(hnsw_base_layer_filename)

    requested_metrics = [f"Recall@{k}", "Latency", "qps"]

    flatnav_index.set_num_threads(1)
    metrics = compute_metrics(
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
        requested_metrics=requested_metrics,
    )

    logging.info(f"Metrics = {metrics}")

    # Get edge access distribution during seaerch
    edge_access_distribution: Dict[
        Tuple[int, int], int
    ] = flatnav_index.get_edge_access_counts()
    return edge_access_distribution


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

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy").astype(np.float32, copy=False),
        np.load(f"{base_path}/{dataset_name}.test.npy").astype(np.float32, copy=False),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy").astype(np.uint32, copy=False),
    )


def select_p90_edges(
    edge_access_distribution: Dict[Tuple[int, int], int], percentile: float = 90
) -> List[Tuple[int, int]]:
    """
    Select a subset of the edges from the distribution that fall above the 90th
    percentile of edge access counts. Return this subset.
    :param edge_access_distribution: The edge access distribution to consider.
    :return selected_edges: The subset of edges that fall above the 90th percentile.
    """
    access_counts = list(edge_access_distribution.values())
    p90_threshold = np.percentile(access_counts, percentile)
    selected_edges = [
        edge
        for edge, count in edge_access_distribution.items()
        if count >= p90_threshold
    ]

    return selected_edges


def depth_first_search_iterative(
    start_node: int, adjacency_list: dict, visited: set
) -> None:
    stack = [start_node]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(
                [
                    neighbor
                    for neighbor in adjacency_list.get(node, [])
                    if neighbor not in visited
                ]
            )


def compute_connected_components(selected_edges: list, outdegree_table: list) -> int:
    adjacency_list = {}
    for edge in selected_edges:
        src, dst = edge
        if src not in adjacency_list:
            adjacency_list[src] = []
        if dst not in adjacency_list:
            adjacency_list[dst] = []
        adjacency_list[src].append(dst)

    visited = set()
    connected_components = 0

    for node in adjacency_list.keys():
        if node not in visited:
            depth_first_search_iterative(node, adjacency_list, visited)
            connected_components += 1

    return connected_components


def analyze_edge_traversals_with_graphtool(
    edge_access_distribution: dict, outdegree_table: List[List[int]]
) -> None:
    from graph_tool.all import Graph, graph_draw

    # Create a directed graph
    g = Graph(directed=True)

    # Ensure there is a vertex for each node (assuming nodes are numbered from 0 to max index)
    max_node = max(
        max(outdegree) for outdegree in outdegree_table if outdegree
    )  # Find the highest node number
    g.add_vertex(max_node + 1)  # Add vertices (graph-tool indexes vertices from 0)

    # Add edges to the graph
    for node, outdegrees in enumerate(outdegree_table):
        for neighbor in outdegrees:
            g.add_edge(g.vertex(node), g.vertex(neighbor))

    # Visualize the graph
    # `graph_draw` is a versatile function for drawing graphs, with many options to customize the visualization.
    graph_draw(
        g,
        vertex_text=g.vertex_index,
        vertex_font_size=18,
        output_size=(1000, 1000),
        output="graph.png",
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

        # Load the edge access distribution from the JSON file
        # with open(f"{dataset_name}_edge_traversal.json", "r") as f:
        #     edge_access_counts = json.load(f)

        # edge_access_counts = main(
        #     dataset_name=dataset_name,
        #     train_dataset=train_dataset,
        #     queries=queries,
        #     ground_truth=ground_truth,
        #     distance_type=metric,
        #     max_edges_per_node=args.num_node_links,
        #     ef_construction=args.ef_construction,
        #     ef_search=args.ef_search,
        #     k=args.k,
        # )

        # Save the edge access distribution to a file for later analysis. Save it using pickle
        # with open(f"{dataset_name}_edge_traversal.pkl", "wb") as f:
        #     pickle.dump(edge_access_counts, f)

        # Save the outdegree table to a file for later analysis. Save it using pickle
        # with open(f"{dataset_name}_outdegree_table.pkl", "wb") as f:
        #     pickle.dump(outdegree_table, f)

        # Load the outdegree table from the file
        with open(f"{dataset_name}_outdegree_table.pkl", "rb") as f:
            outdegree_table = pickle.load(f)

        # Load the edge access distribution from the file
        with open(f"{dataset_name}_edge_traversal.pkl", "rb") as f:
            edge_access_counts = pickle.load(f)

        selected_edges = select_p90_edges(edge_access_counts, percentile=99)

        len_ = len(selected_edges)
        print(f"Number of edges = {len_}")

        connected_components = compute_connected_components(
            selected_edges=selected_edges, outdegree_table=outdegree_table
        )
        print(f"Connected components = {connected_components}\n")

        # analyze_edge_traversals_with_graphtool(
        #     edge_access_distribution=edge_access_counts, outdegree_table=outdegree_table
        # )
