from copy import deepcopy
import plotly.express as px
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import re
import seaborn as sns


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


def plot_metrics_by_similarity(metrics: dict, similarity: str):
    # Filter the datasets based on the similarity type (cosine or l2)
    filter_cosine = [name.endswith(similarity) for name in metrics["dataset_names"]]
    df_hnsw = pd.DataFrame(
        {
            "Skewness": [
                skew for skew, f in zip(metrics["skewness_hnsw"], filter_cosine) if f
            ],
            "Latency": [
                lat for lat, f in zip(metrics["latency_hnsw"], filter_cosine) if f
            ],
            "Algorithm": "HNSW",
            "Dataset": [
                name for name, f in zip(metrics["dataset_names"], filter_cosine) if f
            ],
        }
    )
    df_flatnav = pd.DataFrame(
        {
            "Skewness": [
                skew for skew, f in zip(metrics["skewness_flatnav"], filter_cosine) if f
            ],
            "Latency": [
                lat for lat, f in zip(metrics["latency_flatnav"], filter_cosine) if f
            ],
            "Algorithm": "FlatNav",
            "Dataset": [
                name for name, f in zip(metrics["dataset_names"], filter_cosine) if f
            ],
        }
    )
    df = pd.concat([df_hnsw, df_flatnav], ignore_index=True)

    # Create the scatter plot with larger marker size and line width
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        symbol="Algorithm",
        size_max=15,
        hover_name="Dataset",
        title=f"Mean query latency vs hubness score ({similarity})",
        width=1200,  # Width of the figure in pixels
        height=800,  # Height of the figure in pixels
    )

    # Get colors for each algorithm from the scatter plot
    colors = {
        alg: fig.data[idx].marker.color
        for idx, alg in enumerate(df["Algorithm"].unique())
    }

    # Add line traces for each algorithm with increased line width
    for algorithm, color in colors.items():
        df_alg = df[df["Algorithm"] == algorithm].sort_values(by="Skewness")
        fig.add_trace(
            go.Scatter(
                x=df_alg["Skewness"],
                y=df_alg["Latency"],
                mode="lines+markers",
                name=algorithm,
                line=dict(color=color, width=4),  # Increase the width of the line
                marker=dict(size=10),  # Increase marker size
                showlegend=False,
            )
        )

    # Update layout to increase font size and adjust legend position
    fig.update_layout(
        legend_title_text="Algorithm",
        xaxis_title="Hubness Score",
        yaxis_title="Latency (ms)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.95,  # Adjust the vertical position
            xanchor="left",
            x=0.01,  # Adjust the horizontal position
        ),
        title_font_size=30,  # Increase title font size
        font=dict(size=20),  # Increase font size for axis titles and ticks
    )
    fig.show()
    html_file_name = f"hubness_{similarity}.html"
    fig.write_html(html_file_name)


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
    fig.write_html("hubness__.html")


def plot_faceted_grid(metrics: dict):
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
    df = pd.concat([df_hnsw, df_flatnav])

    # Create the faceted grid
    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        color="Algorithm",
        facet_col="Algorithm",
        hover_name="Dataset",
    )
    fig.show()


if __name__ == "__main__":
    # load metrics from JSON
    with open("metrics.json", "r") as f:
        metrics = json.load(f)

    plot_metrics_by_similarity(metrics, "cosine")
    plot_metrics_by_similarity(metrics, "l2")
