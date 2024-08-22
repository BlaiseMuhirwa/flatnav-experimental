import argparse
import flatnav
import numpy as np
from typing import Tuple
import time
import os
import logging
import boto3

logging.basicConfig(level=logging.INFO)
ROOT_DATASET_PATH = "/root/data/"

os.environ["AWS_PROFILE"] = "s3-bucket-reader-writer"


def upload_file_to_s3(
    s3_client: boto3.client, file_name: str, bucket_name: str, bucket_prefix: str
):
    """
    Uploads a file to an S3 bucket.
    :param s3_client: The S3 client to use.
    :param file_name: The file to upload.
    :param bucket_name: The name of the bucket to upload to.
    :param bucket_prefix: The prefix to add to the file in the bucket.
    """
    if not bucket_prefix.endswith("/"):
        bucket_prefix += "/"

    object_name = f"{bucket_prefix}{os.path.basename(file_name)}"
    logging.info(f"Attempting to upload file {file_name} to {object_name}")

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)

    except Exception:
        logging.error(
            f"Failed to upload file {file_name} to {object_name}", exc_info=True
        )
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FlatNav index")

    parser.add_argument(
        "--dataset-names",
        type=str,
        required=True,
        nargs="+",
        help="dataset names. All will be expected to be at the same path.",
    )

    parser.add_argument(
        "--ef-construction", type=int, required=True, help="ef-construction parameter."
    )

    parser.add_argument(
        "--bucket-name",
        type=str,
        required=True,
        help="Name of the S3 bucket to store the index.",
    )

    args = parser.parse_args()
    return args


def build_and_push_to_s3(
    s3_client: "botocore.client.S3",
    bucket_name: str,
    dataset_name: str,
    train_dataset: np.ndarray,
    distance_type: str,
    max_edges_per_node: int,
    ef_construction: int,
    num_initializations: int = 100,
) -> Tuple[dict, dict]:
    """

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

    # Build FlatNav index and configure it to perform search by using random initialization
    flatnav_index = flatnav.index.index_factory(
        distance_type=distance_type,
        dim=dim,
        dataset_size=dataset_size,
        max_edges_per_node=max_edges_per_node,
        verbose=True,
        # collect_stats=True,
        # use_random_initialization=True,
        # random_seed=42,
    )

    flatnav_index.set_num_threads(os.cpu_count())

    # Train the index.
    start = time.time()
    flatnav_index.add(
        data=train_dataset,
        ef_construction=ef_construction,
        num_initializations=num_initializations,
    )
    end = time.time()
    logging.info(f"Index construction time: {end - start:.3f} seconds")

    # Save index to disk then upload to s3
    index_filename = f"index_{dataset_name}.index"
    flatnav_index.save(index_filename)
    upload_file_to_s3(
        s3_client=s3_client,
        file_name=index_filename,
        bucket_name=bucket_name,
        bucket_prefix=f"{dataset_name}",
    )
    
    os.remove(index_filename)



def load_dataset(base_path: str, dataset_name: str) -> np.ndarray:
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Dataset path not found at {base_path}")
    return (
        np.load(f"{base_path}/{dataset_name}.train.npy")
        # np.load(f"{base_path}/{dataset_name}.test.npy"),
        # np.load(f"{base_path}/{dataset_name}.gtruth.npy"),
    )
    
def get_metric_from_dataset_name(dataset_name: str) -> str:
    """
    Extract the metric from the dataset name. The metric is the last part of the dataset name.
    Ex. normal-10-euclidean -> l2
        mnist-784-euclidean -> l2
        normal-10-angular -> angular
    """
    _, _, metric = dataset_name.split("-")
    if metric == "euclidean":
        return "l2"
    elif metric == "angular":
        return "angular"
    raise ValueError(f"Invalid metric: {metric}")


def main():
    args = parse_args()
    
    s3_client = boto3.client("s3")

    for dataset_name in args.dataset_names:
        base_path = os.path.join(ROOT_DATASET_PATH, dataset_name)
        train_dataset = load_dataset(base_path, dataset_name)

        logging.info(f"Saving dataset to S3...")

        upload_file_to_s3(
            s3_client=s3_client,
            file_name=f"{base_path}/{dataset_name}.train.npy",
            bucket_name=args.bucket_name,
            bucket_prefix=dataset_name,
        )

        upload_file_to_s3(
            s3_client=s3_client,
            file_name=f"{base_path}/{dataset_name}.test.npy",
            bucket_name=args.bucket_name,
            bucket_prefix=dataset_name,
        )

        upload_file_to_s3(
            s3_client=s3_client,
            file_name=f"{base_path}/{dataset_name}.gtruth.npy",
            bucket_name=args.bucket_name,
            bucket_prefix=dataset_name,
        )
        
        metric_name = get_metric_from_dataset_name(dataset_name)

        build_and_push_to_s3(
            s3_client=s3_client,
            bucket_name=args.bucket_name,
            dataset_name=dataset_name,
            train_dataset=train_dataset,
            distance_type=metric_name,
            max_edges_per_node=32,
            ef_construction=args.ef_construction,
        )
        
if __name__=="__main__":
    main()
