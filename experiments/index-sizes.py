import sys
import json
import flatnav.index
import numpy as np
import argparse
import os
from typing import Tuple, Union
import logging
import hnswlib
from memray import Tracker

logging.basicConfig(level=logging.INFO)

ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute hubness of a dataset")

    parser.add_argument(
        "--datasets",
        type=str,
        required=True,
        nargs="+",
        help="dataset names. All will be expected to be at theh same path.",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        required=True,
        help="Index type to use (flatnav or hnsw)",
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
        "--num-node-links",
        type=int,
        required=True,
        help="max-edges-per-node parameter.",
    )

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> np.ndarray:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return np.load(f"{base_path}/{dataset_name}.train.npy").astype(
        np.float32, copy=False
    )


def build_index(
    index_type: str,
    train_dataset: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
):
    dataset_size, dim = train_dataset.shape

    if index_type.lower() == "flatnav":
        index = flatnav.index.index_factory(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=False,
        )
        index.set_num_threads(os.cpu_count())

        index.add(
            data=train_dataset, ef_construction=ef_construction, num_initializations=100
        )

    elif index_type.lower() == "hnsw":
        index = hnswlib.Index(
            space=distance_type if distance_type == "l2" else "ip", dim=dim
        )
        index.init_index(
            max_elements=dataset_size,
            ef_construction=ef_construction,
            M=max_edges_per_node // 2,
        )

        index.set_num_threads(os.cpu_count())
        index.add_items(data=train_dataset, ids=np.arange(dataset_size))

    else:
        raise ValueError(f"Unknown index type {index_type}")

    return index


if __name__ == "__main__":
    args = parse_args()

    dataset_names = args.datasets
    distance_types = args.metrics
    index_type = args.index_type

    index_sizes = {}

    for index, dataset_name in enumerate(dataset_names):
        metric = distance_types[index]
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        train_dataset = load_dataset(base_path=base_path, dataset_name=dataset_name)

        with Tracker("hnsw_mem_profile.bin") as tracker:
            index = build_index(
                index_type=index_type,
                train_dataset=train_dataset,
                distance_type=metric,
                max_edges_per_node=args.num_node_links,
                ef_construction=args.ef_construction,
            )
