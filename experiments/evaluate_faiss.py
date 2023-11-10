import json
import numpy as np
from typing import Optional, Tuple, List
import numpy as np
from dvclive import Live
import faiss
import os
import time
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


def parse_arguments() -> argparse.Namespace:
    """
    Parses arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--datasets", required=True, nargs="+", help="ANNS benchmark dataset to run on."
    )

    parser.add_argument(
        "--pq_m",
        required=True,
        nargs="+",
        type=int,
        help="Number of subquantizers for PQ. This should exactly divide the dataset dimensions.",
    )

    parser.add_argument(
        "--ef_cons",
        required=True,
        nargs="+",
        type=int,
        help="ef_construction. HNSW hyperparameter.",
    )

    parser.add_argument(
        "--ef_search",
        required=True,
        nargs="+",
        type=int,
        help="ef_search. HNSW hyperparameter.",
    )

    parser.add_argument(
        "--num_node_links",
        required=True,
        nargs="+",
        type=int,
        help="Maximum number of edges per node in the graph. HNSW hyperparameter.",
    )

    args = parser.parse_args()
    return args


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


def compute_metrics(
    index, queries: np.ndarray, ground_truth: np.ndarray, k=100
) -> Tuple[float, float]:
    """
    Compute recall and QPS for given queries, ground truth, and a Faiss index.

    Args:
        - index: The Faiss index to search.
        - queries: The query vectors.
        - ground_truth: The ground truth indices for each query.
        - k: Number of neighbors to search.

    Returns:
        Mean recall over all queries.
        QPS (queries per second)
    """
    start = time.time()
    _, top_k_indices = index.search(queries, k)
    end = time.time()

    querying_time = end - start
    qps = len(queries) / querying_time

    # Convert each ground truth list to a set for faster lookup
    ground_truth_sets = [set(gt) for gt in ground_truth]

    mean_recall = 0

    for idx, k_neighbors in enumerate(top_k_indices):
        query_recall = sum(
            1 for neighbor in k_neighbors if neighbor in ground_truth_sets[idx]
        )
        mean_recall += query_recall / k

    recall = mean_recall / len(queries)
    return recall, qps


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
    pq_m_params: List[int],
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    dvc_live_path: str,
):
    # Ensure that the directory exists for dvclive
    os.makedirs(dvc_live_path, exist_ok=True)

    with Live(dvc_live_path) as live:
        for param_key, param_val in ENVIRONMENT_INFO.items():
            live.log_param(param_key, param_val)

        for pq_m in pq_m_params:
            for node_links in num_node_links:
                for ef_cons in ef_cons_params:
                    for ef_search in ef_search_params:
                        live.log_param("node_links", node_links)
                        live.log_param("ef_construction", ef_cons)
                        live.log_param("ef_search", ef_search)
                        live.log_param("pq_m", pq_m)

                        index = train_hnsw_index(
                            data=train_dataset,
                            pq_m=pq_m,
                            num_node_links=node_links,
                            ef_construction=ef_cons,
                            ef_search=ef_search,
                        )
                        recall, qps = compute_metrics(
                            index=index, queries=queries, ground_truth=gtruth
                        )
                        live.log_metric("Recall@100", recall)
                        live.log_metric("QPS", qps)
                        logging.info(
                            f"Recall@100: {recall}, qps={qps}, pq_m={pq_m}, node_links={node_links},"
                            f" ef_cons={ef_cons}, ef_search={ef_search}"
                        )

                        live.next_step()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parse_arguments()

    for dataset in args.datasets:
        train_data, queries, ground_truth = load_benchmark_dataset(dataset_name=dataset)

        pq_m = args.pq_m
        dvc_live_path = f"{dataset}/dvclive"
        main(
            train_dataset=train_data,
            queries=queries,
            gtruth=ground_truth,
            pq_m_params=args.pq_m,
            ef_cons_params=args.ef_cons,
            ef_search_params=args.ef_search,
            num_node_links=args.num_node_links,
            dvc_live_path=dvc_live_path,
        )
