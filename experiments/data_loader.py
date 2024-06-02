import numpy as np
from abc import ABC, abstractmethod
import os
from typing import Tuple, List, Optional, Union


class DatasetLoader(ABC):
    def __init__(
        self,
        train_dataset_path: str,
        queries_path: str,
        ground_truth_path: str,
        range: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Load benchmark dataset, queries and ground truth.
        :param train_dataset_path: Path to the train dataset
        :param queries_path: Path to the queries
        :param ground_truth_path: Path to the ground truth
        :param range: Number of elements to load from the dataset.
                If a tuple is provided, the first element is the start index and
                the second element is the end index
        NOTE: The range parameter will only chunk the training dataset.

        """
        self.verify_paths([train_dataset_path, queries_path, ground_truth_path])
        self.train_dataset_path = train_dataset_path
        self.queries_path = queries_path
        self.ground_truth_path = ground_truth_path

        if range:
            invalid_length = len(range) != 2
            invalid_values = range[0] < 0 or range[1] < 0 or range[0] > range[1]
            if invalid_length or invalid_values:
                raise ValueError(f"Invalid range specified: {range}")

        self.range = range

    def verify_paths(self, paths: List[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found")

    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass


class NpyDatasetLoader(DatasetLoader):
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.range:
            start_index, end_index = self.range
            train_dataset = np.load(self.train_dataset_path)[
                start_index:end_index
            ]
        else:
            train_dataset = np.load(self.train_dataset_path)
        queries = np.load(self.queries_path)
        ground_truth = np.load(self.ground_truth_path)
        return train_dataset, queries, ground_truth


class BvecsDatasetLoader(DatasetLoader):
    """
    This is adapted from the Matlab code provided by the authors of the SIFT dataset.
    Reference:
        1. http://corpus-texmex.irisa.fr/ivecs_read.m
        2. http://corpus-texmex.irisa.fr/bvecs_read.m
    NOTE: This is mostly for loading the SIFT1B dataset.
    """

    def _read_ivecs_file(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            dimension = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec_size = 4 + dimension * 4

            f.seek(0, 2)
            total_vectors = f.tell() // vec_size
            start, end = 1, total_vectors

            if self.range:
                start, end = self.range
                end = min(end, total_vectors)

            assert 1 <= start <= end <= total_vectors, "Invalid range specified."

            f.seek((start - 1) * vec_size, 0)
            v = np.fromfile(
                f, dtype=np.int32, count=(dimension + 1) * (end - start + 1)
            )
            return v.reshape((end - start + 1, dimension + 1))[:, 1:]

    def _read_bvecs_file(self, filename: str) -> np.ndarray:
        with open(filename, "rb") as f:
            dimension = np.fromfile(f, dtype=np.int32, count=1)[0]
            vec_size = 4 + dimension

            f.seek(0, 2)
            total_vectors = f.tell() // vec_size

            start, end = 1, total_vectors
            if self.range:
                start, end = self.range
                end = min(end, total_vectors)

            assert 1 <= start <= end <= total_vectors, "Invalid range specified."

            f.seek((start - 1) * vec_size, 0)
            v = np.fromfile(
                f, dtype=np.uint8, count=(dimension + 4) * (end - start + 1)
            )
            return v.reshape((end - start + 1, dimension + 4))[:, 4:]

    def load_data(self) -> Tuple[np.ndarray]:
        ground_truth = self._read_ivecs_file(self.ground_truth_path)
        # Ground truth has shape (10000, 1000) but we only need the first 100 queries
        ground_truth = ground_truth[:, 0:100]

        train_data = self._read_bvecs_file(self.train_dataset_path)
        queries_data = self._read_bvecs_file(self.queries_path)

        return train_data, queries_data, ground_truth


class FbinDatasetLoader(DatasetLoader):
    """
    NOTE: This is mostly for loading the DEEP1B dataset.
    Uses numpy's memmap to map the file to memory, which is useful for large files like the DEEP1B dataset.
    """

    def load_ground_truth(self, path: str) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """
        Load the IDs and the distances of the top-k's and not the distances.
        Returns:
            - Array of top k IDs
            - Array of top k distances
            - Number of queries
            - K value
        """
        with open(path, "rb") as f:
            num_queries = np.fromfile(f, dtype=np.uint32, count=1)[0]
            K = np.fromfile(f, dtype=np.uint32, count=1)[0]

        # Memory-map the IDs only
        ground_truth_ids = np.memmap(
            path,
            dtype=np.uint32,
            mode="r",
            shape=(num_queries, K),
            offset=8,
        )

        ground_truth_dists = np.memmap(
            path,
            dtype=np.float32,
            mode="r",
            shape=(num_queries, K),
            offset=8 + (num_queries * K * np.dtype(np.uint32).itemsize),
        )

        return ground_truth_ids, ground_truth_dists, num_queries, K

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Read header information (num_points and num_dimensions)
        with open(self.train_dataset_path, "rb") as f:
            num_points = np.fromfile(f, dtype=np.uint32, count=1)[0]
            num_dimensions = np.fromfile(f, dtype=np.uint32, count=1)[0]

        if self.range:
            start_index, end_index = self.range
            if start_index < 0 or end_index > num_points or start_index >= end_index:
                raise ValueError("Invalid range specified.")

            chunk_size = end_index - start_index

            # Calculate the bytes offset from the start of the file data (after header)
            # Each point consists of num_dimensions floats, and each float is np.float32.itemsize (4) bytes
            offset = 8 + start_index * num_dimensions * np.dtype(np.float32).itemsize

            # Using np.memmap to map the chunk of the file
            train_dataset = np.memmap(
                self.train_dataset_path,
                dtype=np.float32,
                mode="r",
                offset=offset,
                shape=(chunk_size, num_dimensions),
            )
        else:
            train_dataset = np.fromfile(
                self.train_dataset_path, dtype=np.float32, offset=8
            )
            train_dataset = train_dataset.reshape((num_points, num_dimensions))

        ground_truth, _, num_queries, _ = self.load_ground_truth(self.ground_truth_path)
        queries_dataset = np.fromfile(
            self.queries_path,
            dtype=np.float32,
            offset=8,
        )
        queries_dataset = queries_dataset.reshape((num_queries, num_dimensions))

        return train_dataset, queries_dataset, ground_truth


def get_data_loader(**kwargs) -> DatasetLoader:
    """
    Factory function for creating the appropriate dataset loader based on the extension of the training dataset.
    :param kwargs: Dictionary of parameters. These must correspond to the parameters of the DatasetLoader constructor.
    """
    train_dataset_path = kwargs.get("train_dataset_path")
    if not train_dataset_path:
        raise ValueError("The train_dataset_path parameter is required.")

    if train_dataset_path.endswith(".npy"):
        return NpyDatasetLoader(**kwargs)
    elif train_dataset_path.endswith(".bvecs"):
        return BvecsDatasetLoader(**kwargs)
    elif train_dataset_path.endswith(".fbin"):
        return FbinDatasetLoader(**kwargs)

    raise ValueError("Invalid file extension for the training dataset.")
