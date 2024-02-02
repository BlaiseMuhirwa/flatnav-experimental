import json
import flatnav.index
import numpy as np
import argparse
import os
from typing import Tuple, Union, Optional
import logging
import hnswlib
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px
from utils import compute_metrics

logging.basicConfig(level=logging.DEBUG)


# ROOT_DATASET_PATH = "/root/data/"
ROOT_DATASET_PATH = os.path.join(os.getcwd(), "..", "data")


def aggregate_metrics(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    ef_search: int,
    k: int,
    search_batch_size: Optional[int] = None,
    num_initializations: int = 100,
) -> Tuple[dict, dict]:
    """
    Computes the following metrics for FlatNav and HNSW:
        - Recall@k
        - Latency
        - QPS
        - Hubness score as measured by the skewness of the k-occurence distribution (N_k)

    NOTE: Index construction is done in parallel, but search is single-threaded.

    :param train_dataset: The dataset to compute the skewness for.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param distance_type: The distance type to use for computing the skewness.
    :param max_edges_per_node: The maximum number of edges per node.
    :param ef_construction: The ef-construction parameter.
    :param ef_search: The ef-search parameter.
    :param k: The number of nearest neighbors to find.
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

    logging.info(f"Building index...")
    hnsw_base_layer_filename = "hnsw_base_layer.mtx"
    hnsw_index.add_items(data=train_dataset, ids=np.arange(dataset_size))
    hnsw_index.save_base_layer_graph(filename=hnsw_base_layer_filename)

    # Build FlatNav index
    logging.info(f"Building FlatNav index...")
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        mtx_filename=hnsw_base_layer_filename,
    )

    logging.info(f"Allocating nodes and building graph links...")
    # Here we will first allocate memory for the index and then build edge connectivity
    # using the HNSW base layer graph. We do not use the ef-construction parameter since
    # it's assumed to have been used when building the HNSW base layer.
    flatnav_index.allocate_nodes(data=train_dataset).build_graph_links()

    # Now delete the HNSW base layer graph since we don't need it anymore
    os.remove(hnsw_base_layer_filename)

    flatnav_index.set_num_threads(num_threads=1)
    hnsw_index.set_num_threads(num_threads=1)

    requested_metrics = [f"recall@{k}", "latency", "qps"]

    logging.info(f"Computing FlatNav metrics...")
    flatnav_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=flatnav_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
        batch_size=search_batch_size,
    )

    logging.info(f"Computing HNSW metrics...")
    hnsw_metrics: dict = compute_metrics(
        requested_metrics=requested_metrics,
        index=hnsw_index,
        queries=queries,
        ground_truth=ground_truth,
        ef_search=ef_search,
        k=k,
        batch_size=search_batch_size,
    )

    logging.info(f"Metrics: Flatnav: {flatnav_metrics}, \n HNSW: {hnsw_metrics}")

    return flatnav_metrics, hnsw_metrics


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
        "--hubness-scores",
        type=str,
        required=True,
        help="JSON file containing hubness scores for each dataset",
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

    parser.add_argument(
        "--search-batch-size",
        type=int,
        default=None,
        required=False,
        help="The number of queries to search in a batch.",
    )

    return parser.parse_args()


def load_dataset(base_path: str, dataset_name: str) -> Tuple[np.ndarray]:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy").astype(copy=False),
        np.load(f"{base_path}/{dataset_name}.test.npy").astype(copy=False),
        np.load(f"{base_path}/{dataset_name}.gtruth.npy").astype(copy=False),
    )


def plot_metrics_seaborn(metrics: dict, k: int):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )

    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )

    # Combine both DataFrames into one for plotting
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Set the style and context for the plot
    sns.set(style="whitegrid", context="talk")
    f, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x="Skewness",
        y="Latency",
        hue="Algorithm",
        style="Algorithm",
        data=df,
        s=100,
        ax=ax,
    )

    # Annotate each point with dataset name
    for i in range(len(df)):
        ax.text(
            df["Skewness"][i] + 0.5,
            df["Latency"][i] + 0.01,
            df["Dataset"][i],
            horizontalalignment="center",
            size="small",
            color="black",
            weight="normal",
        )

    sns.despine(trim=True, left=True)
    ax.set_title(f"Mean Query Latency vs Hubness score(skewness)")
    ax.legend()

    # Save the figure
    plt.savefig("hubness_seaborn.png")


def plot_metrics_plotly(metrics: dict, k: int):
    df_hnsw = pd.DataFrame(
        {
            "Skewness": metrics["skewness_hnsw"],
            "Latency": metrics["latency_hnsw"],
            "Algorithm": "HNSW",
            "Dataset": metrics["dataset_names"],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": metrics["skewness_flatnav"],
            "Latency": metrics["latency_flatnav"],
            "Algorithm": "FlatNav",
            "Dataset": metrics["dataset_names"],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        symbol="Algorithm",
        size_max=15,
        hover_name="Dataset",  # Shows dataset name on hover
        title="Mean query latency vs hubness score",
    )
    fig.update_layout(
        legend_title_text="Algorithm",
        xaxis_title="Skewness",
        yaxis_title="Latency",
        legend=dict(orientation="h", yanchor="top", y=0.01, xanchor="left", x=0.01),
    )
    fig.show()
    fig.write_html("hubness__.html")
    # fig.write_image("hubness.png")


if __name__ == "__main__":
    args = parse_args()

    # Initialize a metrics dictionary to contain recall values for FlatNav and HNSW and the skewness values for FlatNav and HNSW
    metrics = {
        "latency_flatnav": [],
        "latency_hnsw": [],
        "dataset_names": [],
    }

    # replace every instance of "euclidean" with "l2" and "angular" with "cosine" in the
    # dataset names list and form a dictionary of dataset names
    DATASET_NAMES = {}
    for dataset_name in args.datasets:
        name, dimension, metric = dataset_name.split("-")
        if metric == "euclidean":
            metric = "l2"
        elif metric == "angular":
            metric = "cosine"
        else:
            raise ValueError(f"Invalid metric: {metric}")
        DATASET_NAMES[dataset_name] = f"{name}{dimension}-{metric}"

    hubness_scores = os.getcwd() + "/" + args.hubness_scores
    if not os.path.exists(hubness_scores):
        raise FileNotFoundError(f"Hubness scores file not found at {hubness_scores}")
    
    dataset_names = args.datasets

    for index, dataset_name in enumerate(dataset_names):
        print(f"Processing dataset {dataset_name} ({index + 1}/{len(dataset_names)})")
        _, dimension, metric = dataset_name.split("-")
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)

        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Dataset path not found at {base_path}")

        print(f"Loading dataset {dataset_name}...")
        train_dataset, queries, ground_truth = load_dataset(
            base_path=base_path, dataset_name=dataset_name
        )

        flatnav_metrics, hnsw_metrics = aggregate_metrics(
            train_dataset=train_dataset,
            queries=queries,
            ground_truth=ground_truth,
            distance_type=metric,
            max_edges_per_node=args.num_node_links,
            ef_construction=args.ef_construction,
            ef_search=args.ef_search,
            k=args.k,
            search_batch_size=args.search_batch_size,
        )

        metrics["latency_flatnav"].append(flatnav_metrics["latency"])
        metrics["latency_hnsw"].append(hnsw_metrics["latency"])
        metrics["dataset_names"].append(DATASET_NAMES[dataset_name])

    # Add hubness scores to the metrics dictionary
    with open(hubness_scores, "r") as f:
        hubness_scores = json.load(f)
    metrics["skewness_flatnav"] = [
        hubness_scores[dataset_name] for dataset_name in dataset_names
    ]
    metrics["skewness_hnsw"] = [
        hubness_scores[dataset_name] for dataset_name in dataset_names
    ]

    # Save metrics to a JSON file called metrics.json
    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    # Plot the metrics using seaborn
    plot_metrics_plotly(metrics=metrics, k=args.k)
    # plot_metrics_seaborn(metrics=metrics, k=args.k)
