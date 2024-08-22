import logging
from scipy.stats import mannwhitneyu, ttest_ind
import pickle
import json
import numpy as np
import os
from typing import List, Tuple
import pandas as pd 


# This should be a persistent volume mount.
DISTRIBUTIONS_SAVE_PATH = "/root/node-access-distributions"
METRICS_DIR = "/root/metrics"

SYNTHETIC_DATASETS = [
    "normal-1-angular",
    "normal-1-euclidean",
    "normal-2-angular",
    "normal-2-euclidean",
    "normal-4-angular",
    "normal-4-euclidean",
    "normal-8-angular",
    "normal-8-euclidean",
    "normal-16-angular",
    "normal-16-euclidean",
    "normal-32-angular",
    "normal-32-euclidean",
    "normal-64-angular",
    "normal-64-euclidean",
    "normal-128-angular",
    "normal-128-euclidean",
    "normal-256-angular",
    "normal-256-euclidean",
    "normal-1024-angular",
    "normal-1024-euclidean",
    "normal-1536-angular",
    "normal-1536-euclidean",
]

ANN_DATASETS = [
    "sift-128-euclidean",
    "glove-100-angular",
    "nytimes-256-angular",
    "gist-960-euclidean",
    "yandex-deep-10m-euclidean",
    # "yandex-tti-10m-angular",
    "spacev-10m-euclidean",
]


class HubNodesConnectivityTester:
    """
    Class to test the connectivity of the hub nodes using a hypothesis test.
    Here is how the the test is set up:
    1. Hypotheses:
        - Null hypothesis (H0): Hub nodes are not more connected to each other than to randomly chosen nodes.
        - Alternative hypothesis (H1): Hub nodes are more connected to each other than to randomly chosen nodes.
    2. Identify hub nodes:
        - Select the nodes that fall into the 99th percentile of the node access counts.
    3. Define connectivity:
        - Presence of a direct edge in the graph.
    4. Calculate hub-hub connectivity:
        - For each hub node, calculate the number of hub nodes in its outdegree table.
        This yields a distribution of the number of hub nodes each hub node is connected to.
    5. Calculate random-hub connectivity:
        - For a set of randomly-chosen nodes (same size as the set of hub nodes), count
            the number of hub nodes in their outdegree table.
        This yields a distribution of the number of hub nodes each random node is connected to.
    6. Set up a statistical test:
        a.  - Use a one-sided Mann-Whitney U test to test the null hypothesis. The Mann-Whitney U test
            is a good option because it does not assume normality and it is non-parametric.
            - The null hypothesis is rejected if the p-value is less than 0.05.

    :param outdegree_table_path: The path to the pickle file containing the outdegree table.
    :param node_access_counts_path: The path to the pickle file containing the node access counts.
    :param include_hub_nodes_in_sample: Whether to include the hub nodes in the sample of random nodes.

    """

    def __init__(
        self,
        outdegree_table_path: str,
        node_access_counts_path: str,
        dataset_name: str,
        sample_size: int = 500,
        include_hub_nodes_in_sample: bool = False,
    ):
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.include_hub_nodes_in_sample = include_hub_nodes_in_sample
        self.outdegree_table = self._load_outdegree_table(outdegree_table_path)
        self.node_access_counts = self._load_node_access_counts(node_access_counts_path)

    def _load_outdegree_table(self, outdegree_table_path: str) -> np.ndarray:
        """
        Read the outdegree table from the pickle file.
        """
        if not os.path.exists(outdegree_table_path):
            raise FileNotFoundError(
                f"Outdegree table not found at {outdegree_table_path}"
            )

        with open(outdegree_table_path, "rb") as f:
            return pickle.load(f)

    def _load_node_access_counts(self, node_access_counts_path: str) -> dict:
        """
        Read the node access counts from a JSON file.
        """
        if not os.path.exists(node_access_counts_path):
            raise FileNotFoundError(
                f"Node access counts not found at {node_access_counts_path}"
            )

        with open(node_access_counts_path, "r") as f:
            data = json.load(f)

        return {int(k): int(v) for k, v in data.items()}

    def _select_hub_nodes(self, percentile: float) -> List[int]:
        """
        Select the nodes that fall above the given percentile.
        :param percentile: The percentile to consider.
        :return selected_nodes: The subset of nodes that fall above the given percentile.
        """
        access_counts = list(self.node_access_counts.values())
        threshold = np.percentile(access_counts, percentile)
        selected_nodes = [
            node
            for node, count in self.node_access_counts.items()
            if count >= threshold
        ]
        # If the selected number of nodes is higher than the sample size, randomly select
        # self.sample_size without replacement.
        if len(selected_nodes) > self.sample_size:
            selected_nodes = np.random.choice(selected_nodes, size=self.sample_size, replace=False)

        return selected_nodes

    def _calculate_hub_hub_connections(self, hub_nodes: List[int]) -> List[int]:
        """
        Calculate the number of hub nodes in the outdegree table for each hub node.
        :param hub_nodes: The hub nodes to consider.
        :return hub_hub_connections: The number of hub nodes in the outdegree table for each hub node.
        """
        hub_hub_connections = []
        for node in hub_nodes:
            intersection = set(hub_nodes) & set(self.outdegree_table[node])
            hub_hub_connections.append(len(intersection))
        return hub_hub_connections

    def _calculate_random_hub_connections(self, hub_nodes: List[int]) -> List[int]:
        """
        Calculate the number of hub nodes in the outdegree table for a set of randomly chosen nodes.
        :param hub_nodes: The hub nodes to consider.
        :return random_hub_connections: The number of hub nodes in the outdegree table for each random node.
        """
        graph_nodes = list(self.node_access_counts.keys())
        num_hub_nodes = len(hub_nodes)
        if num_hub_nodes >= len(graph_nodes):
            raise ValueError(
                "Number of hub nodes must be less than the total number of nodes."
            )

        # Sample without replacement n nodes from the graph.
        if self.include_hub_nodes_in_sample:
            random_nodes = np.random.choice(
                graph_nodes, size=num_hub_nodes, replace=False
            )
        else:
            non_hub_nodes = list(set(self.node_access_counts.keys()) - set(hub_nodes))
            if len(non_hub_nodes) < num_hub_nodes:
                raise ValueError(
                    "Not enough non-hub nodes to sample from. "
                    f"Num-hub nodes = {num_hub_nodes}, total nodes = {len(graph_nodes)}"
                )
            random_nodes = np.random.choice(
                non_hub_nodes, size=num_hub_nodes, replace=False
            )

        random_hub_connections = []
        for node in random_nodes:
            intersection = set(hub_nodes) & set(self.outdegree_table[node])
            random_hub_connections.append(len(intersection))
        return random_hub_connections

    def calculate_effect_size(
        self, hub_hub_connections: List[int], random_hub_connections: List[int]
    ) -> float:
        """
        Computes Cohen's D effect size between hub-hub and random-hub.
        """
        n1, n2 = len(hub_hub_connections), len(random_hub_connections)
        mean_difference = np.mean(hub_hub_connections) - np.mean(random_hub_connections)
        pooled_variance = (n1 - 1) * np.std(hub_hub_connections) ** 2 + (
            n2 - 1
        ) * np.std(random_hub_connections) ** 2
        pooled_variance /= n1 + n2 - 2
        effect_size = mean_difference / np.sqrt(pooled_variance + 1e-10)
        return effect_size
    
    def calculate_descriptive_stats(self, connections: List[int]) -> Tuple[float, float, float, float]: 
        """
        Computes the mean, std, median and interquartile range (IQR) of the connections.
        """
        mean = np.mean(connections)
        std = np.std(connections)
        median = np.median(connections)
        q1, q3 = np.percentile(connections, [25, 75])
        iqr = q3 - q1
        return mean, std, median, iqr

    def run_hypothesis_tests(self, percentile: float) -> dict[str, list[float, float]]:
        """
        Run the hypothesis test to determine if hub nodes are more connected to each
        other than to randomly chosen nodes.
        :param percentile: The percentile to consider.
        :return test_results: A dictionary containing the test results.
        """
        hub_nodes = self._select_hub_nodes(percentile=percentile)
        logging.info(f"Num hub nodes = {len(hub_nodes)}, dataset = {self.dataset_name}")

        hub_hub_connections: List[int] = self._calculate_hub_hub_connections(
            hub_nodes=hub_nodes
        )
        random_hub_connections: List[int] = self._calculate_random_hub_connections(
            hub_nodes=hub_nodes
        )
        if len(hub_hub_connections) != len(random_hub_connections):
            raise ValueError(
                "Hub-hub connections and random-hub connections must have the same length."
            )

        # Perform the statistical tests
        # Mann-Whitney U test
        u_statistic, mann_whitney_p_value = mannwhitneyu(
            hub_hub_connections, random_hub_connections, alternative="greater"
        )
        logging.info(
            f"Mann-Whitney U test: (u_stat={u_statistic}, p-value={mann_whitney_p_value})"
        )

        # Two sample t-test. This will be the Welch's t-test since we are not assuming
        # equal vairances.
        t_statistic, ttest_p_value = ttest_ind(
            hub_hub_connections,
            random_hub_connections,
            equal_var=False,
            alternative="greater",
        )
        logging.info(
            f"Two-sample t-test: (t_stat={t_statistic}, p-value={ttest_p_value})\n"
        )

        effect_size = self.calculate_effect_size(hub_hub_connections, random_hub_connections)
        hub_stats = self.calculate_descriptive_stats(hub_hub_connections)
        random_stats = self.calculate_descriptive_stats(random_hub_connections)


        return {
            "mann-whitney-u-statistic": u_statistic,
            "mann-whitney-p-value": mann_whitney_p_value,
            "two-sample-t-statistic": t_statistic,
            "two-sample-t-test-p-value": ttest_p_value,
            "num-hub-nodes": len(hub_nodes),
            "effect_size": effect_size,
            "hub_stats": {
                "mean": hub_stats[0],
                "std": hub_stats[1],
                "median": hub_stats[2],
                "iqr": hub_stats[3],
            },
            "random_stats": {
                "mean": random_stats[0],
                "std": random_stats[1],
                "median": random_stats[2],
                "iqr": random_stats[3],
            },
        }


def run_hypothesis_tests() -> None:
    all_test_results = {}
    save_filename = "hypothesis_tests.json"
    csv_filename = "hypothesis_tests.csv"
    save_filename = os.path.join(METRICS_DIR, save_filename)
    csv_filename = os.path.join(METRICS_DIR, csv_filename)

    for dataset_name in SYNTHETIC_DATASETS + ANN_DATASETS:
        outdegree_table_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_outdegree_table.pkl"
        )
        node_access_counts_path = os.path.join(
            DISTRIBUTIONS_SAVE_PATH, f"{dataset_name}_node_access_counts.json"
        )

        tester = HubNodesConnectivityTester(
            outdegree_table_path=outdegree_table_path,
            node_access_counts_path=node_access_counts_path,
            dataset_name=dataset_name,
            include_hub_nodes_in_sample=True,
        )
        test_results = tester.run_hypothesis_tests(percentile=90)

        # Append the test results to a JSON file containing all the results.
        all_test_results[dataset_name] = test_results
        with open(save_filename, "w") as f:
            json.dump(all_test_results, f, indent=4)

    # Convert JSON to CSV
    with open(save_filename, "r") as f:
        json_data = json.load(f)

    # Flatten the JSON data for CSV conversion
    flat_data = []
    for dataset, metrics in json_data.items():
        flat_record = {"dataset_name": dataset}
        flat_record.update(metrics)
        flat_data.append(flat_record)

    df = pd.DataFrame(flat_data)
    df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    run_hypothesis_tests()
