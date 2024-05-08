import time
from typing import Union
import json
import hnswlib
import numpy as np
from typing import Optional, Tuple, List, Dict
import numpy as np
import os
import logging
import platform, socket, psutil
import argparse
import flatnav
from data_loader import get_data_loader
from plotting.plot import create_plot, create_linestyles
from plotting.metrics import metric_manager


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


def compute_metrics(
    requested_metrics: List[str],
    index: Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index],
    queries: np.ndarray,
    ground_truth: np.ndarray,
    ef_search: int,
    k=100,
) -> Dict[str, float]:
    """
    Compute metrics, possibly including recall, QPS, average per query distance computations,
    and latency percentiles for given queries, ground truth for the given index (FlatNav or HNSW).

    :param requested_metrics: A list of metrics to compute. Options include `recall`, `qps`, `latency_p50`,
        `latency_p95`, `latency_p99`, and `latency_p999`.
    :param index: Either a FlatNav or HNSW index to search.
    :param queries: The query vectors.
    :param ground_truth: The ground truth indices for each query.
    :param ef_search: The size of the dynamic candidate list.
    :param k: Number of neighbors to search.

    :return: Dictionary of metrics.

    """
    is_flatnav_index = type(index) in (flatnav.index.L2Index, flatnav.index.IPIndex)
    latencies = []
    top_k_indices = []
    distance_computations = []

    if is_flatnav_index:
        for query in queries:
            start = time.time()
            _, indices = index.search_single(
                query=query,
                ef_search=ef_search,
                K=k,
                num_initializations=100,
            )
            end = time.time()
            latencies.append(end - start)
            top_k_indices.append(indices)

            # Fetches the total number of distance computations for the last query.
            # and resets the counter.
            query_dis_computations = index.get_query_distance_computations()
            distance_computations.append(query_dis_computations)

    else:
        index.set_ef(ef_search)
        for query in queries:
            start = time.time()
            indices, _ = index.knn_query(data=np.array([query]), k=k)
            end = time.time()
            latencies.append(end - start)
            top_k_indices.append(indices[0])

        else:
            # HNSW aggregates distance computations across all queries.
            distance_computations.append(index.get_distance_computations())

    querying_time = sum(latencies)
    distance_computations = sum(distance_computations)
    num_queries = len(queries)

    # Construct a kwargs dictionary to pass to the metric functions.
    kwargs = {
        "querying_time": querying_time,
        "num_queries": num_queries,
        "latencies": latencies,
        "distance_computations": distance_computations,
        "queries": queries,
        "ground_truth": ground_truth,
        "top_k_indices": top_k_indices,
        "k": k,
    }

    metrics = {}

    for metric_name in requested_metrics:
        try:
            if metric_name in metric_manager.metric_functions:
                metrics[metric_name] = metric_manager.compute_metric(
                    metric_name, **kwargs
                )
        except Exception:
            logging.error(f"Error computing metric {metric_name}", exc_info=True)

    return metrics


def create_and_train_hnsw_index(
    data: np.ndarray,
    space: str,
    dim: int,
    dataset_size: int,
    ef_construction: int,
    max_edges_per_node: int,
    num_threads: int,
    base_layer_filename: Optional[str] = None,
) -> Union[hnswlib.Index, None]:
    """
    Kind of messy to return either a hnswlib.Index or None, but we are doing this so that
    when we use the HNSW base layer to construct a Flatnav index, we don't ever have the
    two indices in memory at the same time. This should help with memory usage.
    """
    hnsw_index = hnswlib.Index(space=space, dim=dim)
    hnsw_index.init_index(
        max_elements=dataset_size, ef_construction=ef_construction, M=max_edges_per_node
    )
    hnsw_index.set_num_threads(num_threads)

    start = time.time()
    hnsw_index.add_items(data=data, ids=np.arange(dataset_size))
    end = time.time()
    logging.info(f"Indexing time = {end - start} seconds")

    if base_layer_filename:
        hnsw_index.save_base_layer_graph(filename=base_layer_filename)
        return None

    return hnsw_index


def train_index(
    train_dataset: np.ndarray,
    distance_type: str,
    dim: int,
    dataset_size: int,
    max_edges_per_node: int,
    ef_construction: int,
    index_type: str = "flatnav",
    use_hnsw_base_layer: bool = False,
    hnsw_base_layer_filename: Optional[str] = None,
    num_build_threads: int = 1,
) -> Union[flatnav.index.L2Index, flatnav.index.IPIndex, hnswlib.Index]:
    """
    Creates and trains an index on the given dataset.
    :param train_dataset: The dataset to train the index on.
    :param distance_type: The distance type to use. Options include "l2" and "angular".
    :param dim: The dimensionality of the dataset.
    :param dataset_size: The number of points in the dataset.
    :param max_edges_per_node: The maximum number of edges per node in the graph.
    :param ef_construction: The size of the dynamic candidate list during construction.
    :param index_type: The type of index to create. Options include "flatnav" and "hnsw".
    :param use_hnsw_base_layer: If set, use HNSW's base layer's connectivity for the Flatnav index.
    :param hnsw_base_layer_filename: Filename to save the HNSW base layer graph to.
    :param num_build_threads: The number of threads to use during index construction.
    :return: The trained index.
    """
    if index_type == "hnsw":
        # We use "angular" instead of "ip", so here we are just converting.
        _distance_type = distance_type if distance_type == "l2" else "ip"
        # HNSWlib will have M * 2 edges in the base layer.
        # So if we want to use M=32, we need to set M=16 here.
        hnsw_index = create_and_train_hnsw_index(
            data=train_dataset,
            space=_distance_type,
            dim=dim,
            dataset_size=dataset_size,
            ef_construction=ef_construction,
            max_edges_per_node=max_edges_per_node // 2,
            num_threads=num_build_threads,
        )

        return hnsw_index

    if use_hnsw_base_layer:
        if not hnsw_base_layer_filename:
            raise ValueError("Must provide a filename for the HNSW base layer graph.")

        _distance_type = distance_type if distance_type == "l2" else "ip"
        create_and_train_hnsw_index(
            data=train_dataset,
            space=_distance_type,
            dim=dim,
            dataset_size=dataset_size,
            ef_construction=ef_construction,
            max_edges_per_node=max_edges_per_node // 2,
            num_threads=num_build_threads,
            base_layer_filename=hnsw_base_layer_filename,
        )

        if not os.path.exists(hnsw_base_layer_filename):
            raise ValueError(f"Failed to create {hnsw_base_layer_filename=}")

        index = flatnav.index.index_factory(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=True,
            collect_stats=True,
        )

        # Here we will first allocate memory for the index and then build edge connectivity
        # using the HNSW base layer graph. We do not use the ef-construction parameter since
        # it's assumed to have been used when building the HNSW base layer.
        index.allocate_nodes(data=train_dataset).build_graph_links(
            mtx_filename=hnsw_base_layer_filename
        )
        os.remove(hnsw_base_layer_filename)

    else:
        index = flatnav.index.index_factory(
            distance_type=distance_type,
            dim=dim,
            dataset_size=dataset_size,
            max_edges_per_node=max_edges_per_node,
            verbose=True,
            collect_stats=True,
        )
        index.set_num_threads(num_build_threads)

        # Train the index.
        start = time.time()
        index.add(
            data=train_dataset, ef_construction=ef_construction, num_initializations=100
        )
        end = time.time()

        logging.info(f"Indexing time = {end - start} seconds")

    return index


def main(
    train_dataset: np.ndarray,
    queries: np.ndarray,
    gtruth: np.ndarray,
    ef_cons_params: List[int],
    ef_search_params: List[int],
    num_node_links: List[int],
    distance_type: str,
    metrics_file: str,
    dataset_name: str,
    requested_metrics: List[str],
    index_type: str = "flatnav",
    use_hnsw_base_layer: bool = False,
    hnsw_base_layer_filename: Optional[str] = None,
    reordering_strategies: List[str] | None = None,
    num_initializations: Optional[List[int]] = None,
    num_build_threads: int = 1,
    num_search_threads: int = 1,
):
    def build_and_run_knn_search(ef_cons: int, node_links: int):
        """
        Build the index and run the KNN search.
        This part is here to ensure that two indices are not in memory at the same time.
        With large datasets, we might get an OOM error.
        """
        index = train_index(
            index_type=index_type,
            train_dataset=train_dataset,
            max_edges_per_node=node_links,
            ef_construction=ef_cons,
            dataset_size=dataset_size,
            dim=dim,
            distance_type=distance_type,
            use_hnsw_base_layer=use_hnsw_base_layer,
            hnsw_base_layer_filename=hnsw_base_layer_filename,
            num_build_threads=num_build_threads,
        )

        if reordering_strategies is not None:
            if type(index) not in (
                flatnav.index.L2Index,
                flatnav.index.IPIndex,
            ):
                raise ValueError("Reordering only applies to the FlatNav index.")
            index.reorder(strategies=reordering_strategies)

        index.set_num_threads(num_search_threads)
        for ef_search in ef_search_params:
            # Extend metrics with computed metrics
            metrics.update(
                compute_metrics(
                    requested_metrics=requested_metrics,
                    index=index,
                    queries=queries,
                    ground_truth=gtruth,
                    ef_search=ef_search,
                )
            )
            logging.info(f"Metrics: {metrics}")

            # Add parameters to the metrics dictionary.
            metrics["distance_type"] = distance_type
            metrics["ef_search"] = ef_search
            all_metrics = {experiment_key: []}

            if os.path.exists(metrics_file) and os.path.getsize(metrics_file) > 0:
                with open(metrics_file, "r") as file:
                    try:
                        all_metrics = json.load(file)
                    except json.JSONDecodeError:
                        logging.error(f"Error reading {metrics_file=}")

            if experiment_key not in all_metrics:
                all_metrics[experiment_key] = []

            all_metrics[experiment_key].append(metrics)
            with open(metrics_file, "w") as file:
                json.dump(all_metrics, file, indent=4)

    dataset_size = train_dataset.shape[0]
    dim = train_dataset.shape[1]

    experiment_key = f"{dataset_name}_{index_type}"

    for node_links in num_node_links:
        metrics = {}
        metrics["node_links"] = node_links

        for ef_cons in ef_cons_params:
            metrics["ef_construction"] = ef_cons

            logging.info(f"Building {index_type=}")
            build_and_run_knn_search(ef_cons=ef_cons, node_links=node_links)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Flatnav on Big ANN datasets."
    )

    parser.add_argument(
        "--index-type",
        default="flatnav",
        help="Type of index to benchmark. Options include `flatnav` and `hnsw`.",
    )

    parser.add_argument(
        "--use-hnsw-base-layer",
        action="store_true",
        help="If set, use HNSW's base layer's connectivity for the Flatnav index.",
    )
    parser.add_argument(
        "--hnsw-base-layer-filename",
        default=None,
        help="Filename to save the HNSW base layer graph to. Please use the .mtx extension for clarity.",
    )

    parser.add_argument(
        "--num-node-links",
        nargs="+",
        type=int,
        default=[16, 32],
        help="Number of node links per node.",
    )

    parser.add_argument(
        "--ef-construction",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500],
        help="ef_construction parameter.",
    )

    parser.add_argument(
        "--ef-search",
        nargs="+",
        type=int,
        default=[100, 200, 300, 400, 500, 1000, 2000, 3000, 4000],
        help="ef_search parameter.",
    )

    parser.add_argument(
        "--num-initializations",
        required=False,
        nargs="+",
        type=int,
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a single ANNS benchmark dataset to run on.",
    )
    parser.add_argument(
        "--queries", required=True, help="Path to a singe queries file."
    )
    parser.add_argument(
        "--gtruth",
        required=True,
        help="Path to a single ground truth file to evaluate on.",
    )
    parser.add_argument(
        "--metric",
        required=True,
        default="l2",
        help="Distance tye. Options include `l2` and `angular`.",
    )

    parser.add_argument(
        "--reordering-strategies",
        required=False,
        nargs="+",
        type=str,
        default=None,
        help="A sequence of graph re-ordering strategies(only applies to FlatNav index)."
        "Options include `gorder` and `rcm`.",
    )

    parser.add_argument(
        "--num-build-threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use during index construction.",
    )

    parser.add_argument(
        "--num-search-threads",
        required=False,
        default=1,
        type=int,
        help="Number of threads to use during search.",
    )

    parser.add_argument(
        "--requested-metrics",
        required=False,
        nargs="+",
        type=str,
        default=[
            "recall",
            "qps",
            "latency_p50",
            "latency_p95",
            "latency_p99",
            "latency_p999",
            "distance_computations",
        ],
    )

    parser.add_argument(
        "--metrics-file",
        required=False,
        default="metrics.json",
        help="File to save metrics to.",
    )

    parser.add_argument(
        "--dataset-name",
        required=True,
        type=str,
        help="Name of the benchmark dataset being used.",
    )

    parser.add_argument(
        "--train-dataset-range",
        required=False,
        default=None,
        nargs="+",
        type=int,
        help="The first element is the start index and the second element is the end index. Must be two integers.",
    )

    return parser.parse_args()


def plot_all_metrics(
    metrics_file_path: str,
    dataset_name: str,
    requested_metrics: List[str],
) -> None:
    with open(metrics_file_path, "r") as file:
        all_metrics = json.load(file)

    linestyles = create_linestyles(unique_algorithms=all_metrics.keys())
    metrics_dir = os.path.dirname(metrics_file_path)

    for y_metric in requested_metrics:
        if y_metric == "recall":
            continue
        x_metric = "recall"
        if y_metric == "distance_computations":
            x_metric = "distance_computations"
            y_metric = "recall"

        plot_name = os.path.join(
            metrics_dir, f"{dataset_name}_{y_metric}_v_{x_metric}.png"
        )

        # Here we are aggregating the metrics across all runs in order
        # to get the x, y pairs. So, we need to create this list of tuples
        experiment_runs = {}
        for experiment_key, metrics in all_metrics.items():
            experiment_runs[experiment_key] = [
                (experiment_key, run[x_metric], run[y_metric]) for run in metrics
            ]

        create_plot(
            experiment_runs=experiment_runs,
            raw=True,
            x_scale="linear",
            y_scale="linear",
            x_axis_metric=x_metric,
            y_axis_metric=y_metric,
            linestyles=linestyles,
            plot_name=plot_name,
        )


def run_experiment():
    # This is the root directory inside the Docker container not the host machine.
    ROOT_DIR = "/root"
    args = parse_arguments()

    data_loader = get_data_loader(
        train_dataset_path=args.dataset,
        queries_path=args.queries,
        ground_truth_path=args.gtruth,
        range=args.train_dataset_range,
    )
    train_data, queries, ground_truth = data_loader.load_data()

    num_initializations = args.num_initializations
    if args.index_type.lower() == "hnsw":
        if num_initializations is not None:
            raise ValueError("HNSW does not support num_initializations.")

    metrics_file_path = os.path.join(ROOT_DIR, "metrics", args.metrics_file)

    main(
        train_dataset=train_data,
        queries=queries,
        gtruth=ground_truth,
        ef_cons_params=args.ef_construction,
        ef_search_params=args.ef_search,
        num_node_links=args.num_node_links,
        distance_type=args.metric.lower(),
        dataset_name=args.dataset_name,
        index_type=args.index_type.lower(),
        use_hnsw_base_layer=args.use_hnsw_base_layer,
        hnsw_base_layer_filename=args.hnsw_base_layer_filename,
        reordering_strategies=args.reordering_strategies,
        num_build_threads=args.num_build_threads,
        num_search_threads=args.num_search_threads,
        metrics_file=metrics_file_path,
        num_initializations=num_initializations,
        requested_metrics=args.requested_metrics,
    )

    plot_all_metrics(
        metrics_file_path=metrics_file_path,
        dataset_name=args.dataset_name,
        requested_metrics=args.requested_metrics,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_experiment()
