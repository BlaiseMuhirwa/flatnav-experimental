import json
import numpy as np
import os
import argparse
from typing import Tuple, List, Dict
import flatnav
import pickle
import logging
import hnswlib
import boto3

logging.basicConfig(level=logging.INFO)

# ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")
ROOT_DATASET_PATH = "/root/data"
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"


def upload_to_s3(outdegree_table: List[List[int]], object_name: str, bucket_name: str):
    """
    Upload the outdegree table to S3.
    :param outdegree_table: The outdegree table to upload.
    :param object_name: The name of the object to store in S3.
    :param bucket_name: The name of the S3 bucket.
    """
    pickled_data = pickle.dumps(outdegree_table)
    s3_client = boto3.client("s3")

    s3_client.put_object(Body=pickled_data, Bucket=bucket_name, Key=object_name)


def download_from_s3(s3_client, object_name: str, bucket_name: str) -> List[List[int]]:
    """
    Download the outdegree table from S3.
    :param object_name: The name of the object to download from S3.
    :param bucket_name: The name of the S3 bucket.
    :return outdegree_table: The outdegree table.
    """
    response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
    pickled_data = response["Body"].read()
    outdegree_table = pickle.loads(pickled_data)
    return outdegree_table


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
    Computes the graph outdegree table for the given dataset.

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

    hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        collect_stats=False,
        use_random_initialization=False,
    )

    flatnav_index.allocate_nodes(data=train_dataset).build_graph_links(
        hnsw_base_layer_filename
    )

    outdegree_table = flatnav_index.get_graph_outdegree_table()
    return outdegree_table


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
        np.load(f"{base_path}/{dataset_name}.train.npy"),
        np.load(f"{base_path}/{dataset_name}.test.npy"),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy"),
    )


def select_p90_nodes(
    node_access_counts: Dict[int, int], percentile: float
) -> List[int]:
    """
    Select the nodes that fall above the 90th percentile.
    :param node_access_counts: The node access counts.
    :param percentile: The percentile to consider.
    :return selected_nodes: The subset of nodes that fall above the 90th percentile.
    """
    access_counts = list(node_access_counts.values())
    threshold = np.percentile(access_counts, percentile)
    selected_nodes = [
        node for node, count in node_access_counts.items() if count >= threshold
    ]
    return selected_nodes


def compute_connected_components(
    hub_nodes: List[int], outdegree_table: List[List[int]]
) -> int:
    """
    Compute the number of connected components in the subgraph.
    :param subgraph: The subgraph to consider.
    :param outdegree_table: The outdegree table.
    :return connected_components: The number of connected components.
    """
    # convert subgraph to a set for O(1) lookup
    hub_nodes_set = set(hub_nodes)
    visited = set()

    def dfs(node: int) -> None:
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)

                for neighbor in outdegree_table[current]:
                    if neighbor in hub_nodes_set and neighbor not in visited:
                        stack.append(neighbor)

    connected_components = 0
    # perform DFS for each node in the subgraph formed by the hub nodes
    for node in hub_nodes:
        if node not in visited:
            dfs(node)
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


if __name__ == "__main__":
    args = parse_args()

    dataset_names = args.datasets

    # Map from dataset name to node access counts
    distributions = {}
    s3_client = boto3.client("s3")

    # Keep track of the computed connected components for each dataset
    results_dict = {}

    for index, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset {dataset_name}...")
        metric = get_metric_from_dataset_name(dataset_name)
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            raise ValueError(f"Dataset path not found at {base_path}")

        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        # Download the outdegree table from S3.
        # outdegree_table = download_from_s3(
        #     s3_client=s3_client,
        #     object_name=f"{dataset_name}/{dataset_name}_outdegree_table.pkl",
        #     bucket_name="hnsw-index-snapshots"
        # )

        outdegree_table = main(
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            queries=queries,
            ground_truth=ground_truth,
            distance_type=metric,
            max_edges_per_node=args.num_node_links,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
        )

        # Upload to s3 teh outdegree table.
        upload_to_s3(
            outdegree_table=outdegree_table,
            object_name=f"{dataset_name}/{dataset_name}_outdegree_table.pkl",
            bucket_name="hnsw-index-snapshots",
        )

        logging.info(f"Number of nodes = {len(outdegree_table)}")

        node_access_dist_file = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
        )
        with open(node_access_dist_file, "r") as f:
            node_access_counts = json.load(f)

        # Convert keys and values to integers
        node_access_counts = {int(k): int(v) for k, v in node_access_counts.items()}

        hub_nodes: List[int] = select_p90_nodes(node_access_counts, 90)
        logging.info(f"Number of hub nodes = {len(hub_nodes)}")

        connected_components = compute_connected_components(
            hub_nodes=hub_nodes, outdegree_table=outdegree_table
        )
        logging.info(
            f"Number of connected components for hubs = {connected_components}"
        )

        # Compute the number of connected components in the rest of the graph (excluding the hub nodes)
        non_hub_nodes = set(range(len(outdegree_table)))  # - set(hub_nodes)
        non_hub_nodes = list(non_hub_nodes)

        entire_graph_connected_components = compute_connected_components(
            hub_nodes=non_hub_nodes, outdegree_table=outdegree_table
        )

        logging.info(
            f"Number of connected components for non-hub nodes = {entire_graph_connected_components}"
        )
        current_results = {
            "num_hub_nodes": len(hub_nodes),
            "hub_nodes_cc": connected_components,
            "entire_graph_cc": entire_graph_connected_components,
            "R_h": connected_components / len(hub_nodes),
            "R_g": entire_graph_connected_components / len(non_hub_nodes),
        }
        results_dict[
            dataset_name.replace("angular", "cosine").replace("euclidean", "l2")
        ] = current_results

        # analyze_edge_traversals_with_graphtool(
        #     edge_access_distribution=edge_access_counts, outdegree_table=outdegree_table
        # )

    # Save the results to a file
    results_file = os.path.join(DISTRIBUTIONS_SAVE_PATH, "connected_components.json")
    with open(results_file, "w") as f:
        json.dump(results_dict, f)
