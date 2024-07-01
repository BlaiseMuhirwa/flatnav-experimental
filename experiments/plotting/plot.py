from typing import Set, Dict, List, Tuple, Optional
import numpy as np
import itertools
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import ticker
from .metrics import metric_manager
from .metrics import MetricConfig


def get_plot_title(x_metric: MetricConfig, y_metric: MetricConfig) -> str:
    up_down = "down" if y_metric.worst_value == float("inf") else "up"
    left_right = "left" if x_metric.worst_value == float("inf") else "right"

    return (
        f"{x_metric.description}-{y_metric.description} tradeoff "
        f"- {up_down} and to the {left_right} is better"
    )


def create_pointset(
    data,
    x_axis_metric_name: Optional[str] = "recall",
    y_axis_metric_name: Optional[str] = "qps",
) -> Tuple[List]:
    """
    Sorts the dataset based on the provided metrics, considering their
    "worst" values to decide the sorting order. It then identifies the Pareto frontier,
    selecting points that represent optimal trade-offs between the two metrics.

    :param data: A list of tuples, each containing an algorithm, its name, and two metric values.
    :param x_axis_metric_name: The name of the metric for the x-axis.
    :param y_axis_metric_name: The name of the metric for the y-axis.

    :return: A tuple of lists: (x_values, y_values, labels) for the Pareto frontier and
            (all_x_values, all_y_values, all_labels) for all data points. Each list in
            the tuple contains the corresponding x or y metric values and labels for the
            algorithms either on the Pareto frontier or in the entire dataset.
    """

    x_metric: MetricConfig = metric_manager.get_metric(x_axis_metric_name)
    y_metric: MetricConfig = metric_manager.get_metric(y_axis_metric_name)

    rev_y = -1 if y_metric.worst_value < 0 else 1
    rev_x = -1 if x_metric.worst_value < 0 else 1

    # Sort data by y-axis metric, then x-axis metric
    # This is necessary to generate the Pareto frontier
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    # Generate Pareto frontier
    x_values, y_values, labels = [], [], []
    all_x_values, all_y_values, all_labels = [], [], []
    last_x = x_metric.worst_value

    # Comparator function to determine if a value is better than the last
    comparator = (
        (lambda xvalue, last_x: xvalue > last_x)
        if last_x < 0
        else (lambda xvalue, last_x: xvalue < last_x)
    )

    for algorithm_name, x_value, y_value in data:
        if not x_value or not y_value:
            continue
        all_x_values.append(x_value)
        all_y_values.append(y_value)
        all_labels.append(algorithm_name)
        if comparator(x_value, last_x):
            last_x = x_value
            x_values.append(x_value)
            y_values.append(y_value)
            labels.append(algorithm_name)

    return x_values, y_values, labels, all_x_values, all_y_values, all_labels


def generate_n_colors(n: int) -> List[Tuple[float]]:
    """
    Generates a list of n unique colors, each represented as an RGBA tuple.

    This function ensures that each generated color is visually distinct from
    the others by maximizing the Euclidean distance in RGB space between them.
    The Alpha (A) value is fixed at 1.0 for full opacity.

    :param n: The number of unique colors to generate.
    :return: A list of RGBA color tuples.

    NOTE:
        - The starting color (0.9, 0.4, 0.4, 1.0) is chosen as a visually pleasing
          shade of red with full opacity, providing a base for further color generation.
        - The `vs` range from 0.3 to 0.9 with 7 steps is chosen to create a balanced
          spectrum of color intensities, avoiding extremes (too dark or too light)
          that may not be visually distinct or appealing when used in certain applications.
    """

    vs = np.linspace(start=0.3, stop=0.9, num=7)
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        # Select the color that is farthest from all the colors we have
        new_color = max(
            itertools.product(vs, vs, vs),
            key=lambda x: min(euclidean(x, c) for c in colors),
        )
        # Add the alpha channel (full opacity)
        colors.append(new_color + (1.0,))

    return colors


def create_linestyles(unique_algorithms: Set[str]) -> Dict[str, Tuple]:
    """
    Maps algorithm names to distinct plotting styles, including color, faded color,
    line style, and marker style.

    For each unique algorithm, generates a unique color and assigns cyclically
    varied line and marker styles. A faded color variant (reduced opacity) is also
    provided for each color.

    Parameters:
    :param unique_algorithms: Names of algorithms to be represented.
    :return: A dictionary mapping algorithm names to a tuple of color (RGBA),
            faded color (RGBA, alpha=0.3), line style, and marker style.

    """

    # Sort the list of algorithm names in reverse alphabetical order to ensure color consistency
    unique_algorithms = sorted(list(unique_algorithms), reverse=True)

    colors = dict(zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))
    linestyles = dict(
        (algorithm, ["--", "-.", "-", ":"][i % 4])
        for i, algorithm in enumerate(unique_algorithms)
    )
    markerstyles = dict(
        (algorithm, ["+", "<", "o", "*", "x"][i % 5])
        for i, algorithm in enumerate(unique_algorithms)
    )
    faded = dict(
        (algorithm, (r, g, b, 0.3)) for algorithm, (r, g, b, _) in colors.items()
    )
    output = dict(
        (
            algorithm,
            (
                colors[algorithm],
                faded[algorithm],
                linestyles[algorithm],
                markerstyles[algorithm],
            ),
        )
        for algorithm in unique_algorithms
    )
    return output


def create_plot(
    experiment_runs: dict,
    raw: bool,
    x_scale: str,
    y_scale: str,
    x_axis_metric: str,
    y_axis_metric: str,
    linestyles: dict,
    plot_name: str,
) -> None:
    handles, plot_labels = [], []
    plt.figure(figsize=(12, 9))

    def mean_y(algorithm):
        """
        Sorting by mean y-value helps aligning plots with labels
        """
        (
            x_values,
            y_values,
            labels,
            all_x_values,
            all_y_values,
            all_labels,
        ) = create_pointset(experiment_runs[algorithm], x_axis_metric, y_axis_metric)
        return -np.log(np.array(y_values)).mean()

    min_x, max_x = 1, 0
    # Find range for logit x-scale
    for algorithm in sorted(experiment_runs.keys(), key=mean_y):
        (
            x_values,
            y_values,
            labels,
            all_x_values,
            all_y_values,
            all_labels,
        ) = create_pointset(experiment_runs[algorithm], x_axis_metric, y_axis_metric)

        min_x = min([min_x] + [x for x in x_values if x > 0])
        max_x = max([max_x] + [x for x in x_values if x < 1])
        color, faded, linestyle, marker = linestyles[algorithm]
        (handle,) = plt.plot(
            x_values,
            y_values,
            "-",
            label=algorithm,
            color=color,
            ms=7,
            mew=3,
            lw=3,
            marker=marker,
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                all_x_values,
                all_y_values,
                label=algorithm,
                color=faded,
                ms=5,
                mew=2,
                lw=2,
                marker=marker,
            )
        plot_labels.append(algorithm)

    x_metric: MetricConfig = metric_manager.get_metric(x_axis_metric)
    y_metric: MetricConfig = metric_manager.get_metric(y_axis_metric)
    ax = plt.gca()
    ax.set_xlabel(x_metric.description)
    ax.set_ylabel(y_metric.description)

    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def func(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_func(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(func, inv_func))
        if alpha <= 3:
            ticks = [inv_func(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        else:
            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])

    else:
        ax.set_xscale(x_scale)

    ax.set_yscale(y_scale)
    ax.set_title(get_plot_title(x_metric, y_metric))
    plt.gca().get_position()
    ax.legend(
        handles,
        plot_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 9},
    )
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    if "range" in x_metric and x_scale != "logit":
        x0, x1 = x_metric.range
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "range" in y_metric:
        plt.ylim(y_metric.range)

    ax.spines["bottom"]._adjust_location()
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close()
