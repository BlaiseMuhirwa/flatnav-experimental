import json 
import os 
import networkx as nx 
import argparse 
import numpy as np
import hnswlib 
import time 
import logging 
from utils import load_graph_from_mtx_file, get_metric_from_dataset_name, load_dataset
from multiprocessing import Pool 

ROOT_DATASET_PATH = "/root/data"
CENTRALITY_SAVE_PATH = "/root/metrics"

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
        "--num-node-links",
        type=int,
        required=True,
        help="max-edges-per-node parameter.",
    )

    return parser.parse_args()

def main(
    dataset_name: str,
    train_dataset: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
):
    """
    :param dataset_name: The name of the dataset.
    :param train_dataset: The dataset to compute the skewness for.
    :param distance_type: The distance type to use for computing the skewness.
    :param max_edges_per_node: The maximum number of edges per node.
    :param ef_construction: The ef-construction parameter.
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

    hnsw_index.set_num_threads(40)

    start = time.time()
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"Indexing time = {end - start} seconds")

    mtx_filename = f"{dataset_name}_hnsw_index.mtx"
    hnsw_index.save_base_layer_graph(filename=mtx_filename)

    # Load the graph from the file and compute the betweenness centrality
    G = load_graph_from_mtx_file(mtx_filename)
    G = nx.DiGraph(G)
    betweenness_centrality = nx.betweenness_centrality(G, normalized=True, endpoints=False)

    # Save the betweenness centrality values to a JSON file with name <dataset_name>_betweenness_centrality.json
    betweenness_centrality_filename = f"{dataset_name}_betweenness_centrality.json"
    betweenness_centrality_filename = os.path.join(CENTRALITY_SAVE_PATH, betweenness_centrality_filename)
    with open(betweenness_centrality_filename, "w") as f:
        json.dump(betweenness_centrality, f)



def run_main(args: argparse.Namespace) -> None:
    dataset_names = args.datasets

    # Map from dataset name to node access counts
    distributions = {}

    # Prepare arguments for multiprocessing
    tasks = []
    for index, dataset_name in enumerate(dataset_names):
        metric = get_metric_from_dataset_name(dataset_name)
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            # Create the directory if it doesn't exist
            raise ValueError(f"Dataset path not found at {base_path}")

        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        task = (
            dataset_name,
            train_dataset,
            metric,
            args.num_node_links,
            args.ef_construction,
        )
        tasks.append(task)

    # Run the tasks in parallel
    with Pool(processes=len(dataset_names)) as pool:
        pool.starmap(main, tasks)

    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_main(parse_args())
