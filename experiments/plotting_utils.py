import plotly.express as px
import pandas as pd


def plot_bubble_chart(metrics: dict):
    # Adding a placeholder for the size of the datasets
    df["DataSize"] = pd.Series(
        [1 for _ in range(len(df))]
    )  # Replace with actual data size values

    fig = px.scatter(
        df,
        x="Skewness",
        y="Latency",
        size="DataSize",  # This is the third variable encoded as bubble size
        color="Algorithm",
        hover_name="Dataset",
    )
    fig.show()


def plot_heatmap(metrics: dict):
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

    # Pivot for heatmap
    heatmap_data = df.pivot_table(
        values="Latency", index="Dataset", columns="Algorithm", aggfunc="mean"
    )

    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Algorithm", y="Dataset", color="Latency"),
        x=["HNSW", "FlatNav"],
    )
    fig.show()


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
