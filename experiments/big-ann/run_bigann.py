import json
import numpy as np
from typing import Optional, Tuple, List
import numpy as np
from dvclive import Live
import faiss
import os
import logging
import platform, socket, psutil
import argparse


ENVIRONMENT_INFO = {
    "load_before_experiment": os.getloadavg()[2],
    "platform": platform.platform(),
    "platform_version": platform.version(),
    "platform_release": platform.release(),
    "architecture": platform.machine(),
    "processor": platform.processor(),
    "hostname": socket.gethostname(),
    "ram_gb": round(psutil.virtual_memory().total / (1024.0**3)),
    "num_cores": psutil.cpu_count(logical=True),
}


DATASETS = {
    "mnist-784-euclidean": (
        "mnist-784-euclidean.train.npy",
        "mnist-784-euclidean.test.npy",
        "mnist-784-euclidean.gtruth.npy",
    ),
    "glove-25-angular": (
        "glove-25-angular.train.npy",
        "glove-25-angular.test.npy",
        "glove-25-angular.gtruth.npy",
    ),
    "glove-100-angular": (
        "glove-100-angular.train.npy",
        "glove-100-angular.test.npy",
        "glove-100-angular.gtruth.npy",
    ),
    "glove-200-angular": (
        "glove-200-angular.train.npy",
        "glove-200-angular.test.npy",
        "glove-200-angular.gtruth.npy",
    ),
    "sift-128-euclidean": (
        "sift-128-euclidean.train.npy",
        "sift-128-euclidean.test.npy",
        "sift-128-euclidean.gtruth.npy",
    ),
}


def load_benchmark_dataset(dataset_name: Optional[None]) -> Tuple[np.ndarray]:
    """
    This assumes that we have a /data/<dataset_name> already present.
    This data directory can be generated by running
        $ /bin/download_anns_datasets.sh <dataset-name>

    This directory will be expected to have the following files:
        - <dataset-name>/<dataset-name>.train.npy
        - <dataset-name>/<dataset-name>.test.npy
        - <dataset-name>/<dataset-name>.gtruth.npy
    """
    dataset_name = dataset_name.lower()
    if not dataset_name in DATASETS.keys():
        raise AssertionError(
            f"{dataset_name=} not in the list of supported datasets."
            "Consider adding it to the list of checking if you misspelled the name."
        )

    train_file, queries_file, gtruth_file = DATASETS[dataset_name]
    base_dir = os.path.join(os.getcwd(), "..", "data", dataset_name)

    return (
        np.load(os.path.join(base_dir, train_file)),
        np.load(os.path.join(base_dir, queries_file)),
        np.load(os.path.join(base_dir, gtruth_file)),
    )


def compute_recall(index, queries: np.ndarray, ground_truth: np.ndarray, k=100):
    """
    Compute recall for given queries, ground truth, and a Faiss index.

    Args:
        - index: The Faiss index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
    """
    _, top_k_indices = index.search(queries, k)

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall


def train_hnsw_index(
    data,
    pq_m,
    num_node_links,
    ef_construction: Optional[int] = 128,
    ef_search: Optional[int] = 128,
    serialize=False,
):
    """
    Train a HNSW index topped with product quantization
    Args:
        - pq_m: Number of subquantizers for PQ. This should exactly divide the
            dataset dimensions.
        - num_node_links: Maximum number of links to keep for each node in the graph
        - serialize: Serialize so we can get a sense of how large the index binary is.

    Returns:
        Index

    Helpful link on correct usage: https://github.com/facebookresearch/faiss/issues/1621
    """
    # configure the index
    dim = data.shape[1]  # data dimension

    # Create the HNSW index
    index = faiss.IndexHNSWPQ(dim, pq_m, num_node_links)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearchh = ef_search

    logging.info("Training index...")
    index.train(data)

    # Add vectors to the index
    index.add(data)

    if serialize:
        # Serialize the index to disk
        buffer = faiss.serialize_index(index)
        index_size = len(buffer)

        logging.info(f"Index size: {index_size} bytes")

    return index

def main(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    gtruth: np.ndarray,
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    pq_m: Optional[int] = 8,
):
    with Live() as live:
        for param_key, param_val in ENVIRONMENT_INFO.items():
            live.log_param(param_key, param_val)

        for node_links in num_node_links:
            for ef_cons in ef_cons_params:
                for ef_search in ef_search_params:
                    live.log_param("node_links", node_links)
                    live.log_param("ef_construction", ef_cons)
                    live.log_param("ef_search", ef_search)

                    index = train_hnsw_index(
                        data=train_dataset,
                        pq_m=pq_m,
                        num_node_links=node_links,
                        ef_construction=ef_cons,
                        ef_search=ef_search,
                    )
                    recall = compute_recall(
                        index=index, queries=queries, ground_truth=gtruth
                    )
                    live.log_metric("Recall@100", recall)
                    logging.info(
                        f"Recall@100: {recall}, node_links={node_links}, ef_cons={ef_cons}, ef_search={ef_search}"
                    )

                    live.next_step()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--datasets", required=True, nargs="+", help="ANNS benchmark dataset to run on."
    )
    parser.add_argument(
        "--log_metrics", required=False, default=False, help="Log metrics to MLFlow."
    )

    args = parser.parse_args()

    ef_constructions = [32, 64, 128]
    ef_searches = [32, 64, 128]
    num_node_links = [8, 16, 32, 64]

    for dataset in args.datasets:
        train_data, queries, ground_truth = load_benchmark_dataset(dataset_name=dataset)

        main(
            train_dataset=train_data,
            queries=queries,
            gtruth=ground_truth,
            ef_cons_params=ef_constructions,
            ef_search_params=ef_searches,
            num_node_links=num_node_links,
        )
