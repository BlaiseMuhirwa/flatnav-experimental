import numpy as np
from sklearn.neighbors import NearestNeighbors
import json
import os
import argparse
import faiss


def compute_k_occurence_distrubution(top_k_indices: np.ndarray) -> np.ndarray:
    """
    Computes the distribution of k-occurences for each node in the given array.
    :param top_k_indices: array of shape (dataset_size, k) containing the indices of
            the k nearest neighbors for each node.

    :return: array of shape (dataset_size,) containing the k-occurence distribution for each node (N_k)
    """

    # validate indices. If any value is negative, throw an error
    if np.any(top_k_indices < 0):
        raise ValueError("Indices cannot be negative")

    dataset_size = top_k_indices.shape[0]
    k_occurence_distribution = np.zeros(dataset_size, dtype=int)

    flattened_indices = top_k_indices.flatten()
    unique_indices, counts = np.unique(flattened_indices, return_counts=True)
    k_occurence_distribution[unique_indices] = counts

    return k_occurence_distribution


def compute_skewness(dataset: np.ndarray, k: int, metric: str) -> float:
    # For cosine distance, we will assume that the data was normalized

    if metric in ["cosine", "angular", "ip"]:
        index = faiss.IndexFlatIP(dataset.shape[1])
    else:
        index = faiss.IndexFlatL2(dataset.shape[1])

    # Shuffle the dataset and add only the first 10k elements to the index
    # np.random.shuffle(dataset)
    # dataset = dataset[0:10000]

    index.add(dataset)
    _, top_k_indices = index.search(dataset, k=k)

    k_occurence_distribution = compute_k_occurence_distrubution(
        top_k_indices=top_k_indices
    )
    mean = np.mean(k_occurence_distribution)
    std_dev = np.std(k_occurence_distribution)
    denominator = len(k_occurence_distribution) * (std_dev**3)
    skewness = (np.sum((k_occurence_distribution - mean) ** 3)) / denominator

    return skewness


if __name__ == "__main__":
    # We will compute the hubness scores for all given datasets and
    # save them in a dictionary as JSON

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", type=str, required=True)
    parser.add_argument("--dataset-names", type=str, nargs="+", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--save-file", type=str, required=True)

    args = parser.parse_args()

    file_path = os.path.join(args.base_path, args.save_file)

    dataset_names = args.dataset_names
    for dataset_name in dataset_names:
        print(f"Computing hubness score for {dataset_name}")

        dataset_path = os.path.join(
            args.base_path, dataset_name, f"{dataset_name}.train.npy"
        )
        _, dimension, metric = dataset_name.split("-")
        metric = metric if metric == "euclidean" else "cosine"

        dataset = np.load(dataset_path)
        dataset = dataset.astype(np.float32, copy=False)

        skewness = compute_skewness(dataset=dataset, k=args.k, metric=metric)
        print(f"Skewness: {skewness}")

        # Read the existing data from the JSON file
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                hubness_scores = json.load(file)
        else:
            hubness_scores = {}

        # Update the dictionary with the new hubness score
        hubness_scores[dataset_name] = skewness

        # Write the updated dictionary back to the JSON file
        with open(file_path, "w") as file:
            json.dump(hubness_scores, file, indent=4)
