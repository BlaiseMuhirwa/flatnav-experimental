import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import faiss


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

    def normalize_rows_inplace(matrix):
        for row in matrix:
            norm = np.linalg.norm(row)
            norm = norm if norm > 0 else 1e-30
            row /= norm

    def add_data_in_batches(index, data, batch_size=10000):
        for i in range(0, data.shape[0], batch_size):
            index.add(data[i : i + batch_size])

    dataset = np.random.normal(size=(num_samples, num_dimensions))
    np.random.shuffle(dataset)
    query_set = dataset[:num_queries]
    dataset_without_queries = dataset[num_queries:]

    if metric in ["cosine", "angular", "ip"]:
        normalize_rows_inplace(dataset_without_queries)
        normalize_rows_inplace(query_set)

        print("Finished normalizing")
        index = faiss.IndexFlatIP(dataset.shape[1])
    else:
        index = faiss.IndexFlatL2(dataset.shape[1])

    add_data_in_batches(index, dataset_without_queries)

    print("kNN search")
    _, ground_truth_labels = index.search(query_set, k=k)

    if np.any(ground_truth_labels < 0):
        raise ValueError("Indices cannot be negative")

    # Create directory if it doesn't exist
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # dataset_without_queries = dataset_without_queries.astype(np.float32, copy=False)
    # query_set = query_set.astype(np.float32, copy=False)
    ground_truth_labels = ground_truth_labels.astype(np.int32, copy=False)

    print("Saving dataset")
    # Save the dataset
    np.save(
        f"{directory_path}/{dataset_name}.train.npy",
        dataset_without_queries,
    )
    np.save(f"{directory_path}/{dataset_name}.test.npy", query_set)
    np.save(
        f"{directory_path}/{dataset_name}.gtruth.npy",
        ground_truth_labels,
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


def check_datasets_exists(base_path: str, dataset_name: str) -> bool:
    train_path = os.path.join(base_path, f"{dataset_name}.train.npy")
    queries = os.path.join(base_path, f"{dataset_name}.test.npy")
    ground_truth = os.path.join(base_path, f"{dataset_name}.gtruth.npy")

    all_exists = all(
        [
            os.path.exists(train_path),
            os.path.exists(queries),
            os.path.exists(ground_truth),
        ]
    )
    return all_exists


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--dataset-size", type=int, required=True)
    parser.add_argument("--num-queries", type=int, required=True)
    parser.add_argument("--dimensions", type=int, nargs="+", required=True)
    parser.add_argument("--k", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    base_path = args.base_path
    dimensions = args.dimensions

    DATASET_NAMES = [f"normal-{d}-angular" for d in dimensions]
    DATASET_NAMES += [f"normal-{d}-euclidean" for d in dimensions]
    # DATASET_NAMES = [f"normal-{d}-euclidean" for d in dimensions]

    # Create the datasets. First create the directory if it doesn't exist
    for dataset_name in DATASET_NAMES:
        directory_path = os.path.join(base_path, dataset_name)

        if check_datasets_exists(directory_path, dataset_name):
            print(f"Dataset {dataset_name} already exists. Skipping...")
            continue
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        print(f"Generating dataset: {dataset_name}")

        _, dimension, metric = dataset_name.split("-")
        metric = metric if metric == "euclidean" else "cosine"
        # Generate the datasets
        generate_iid_normal_dataset(
            num_samples=args.dataset_size,
            num_dimensions=int(dimension),
            num_queries=args.num_queries,
            k=args.k,
            directory_path=directory_path,
            dataset_name=dataset_name,
            metric=metric,
        )
