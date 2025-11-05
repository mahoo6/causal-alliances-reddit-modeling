import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import umap
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", category=UserWarning, module="umap")
warnings.filterwarnings("ignore", message=".*omp_set_nested.*")

def process_embeddings(emb_df: pd.DataFrame):
    """
    Reduces subreddit embeddings to 2D using UMAP and prepares
    a coordinate map for visualization.

    Args:
        emb_df (pd.DataFrame): DataFrame containing subreddit embeddings.
                               Must include 'subreddit' and embedding columns.

    Returns:
        - emb_df: DataFrame with added columns ['x', 'y']
        - emb_map: dict{subreddit: {'x': x, 'y': y}} for quick coordinate access
    """
    vec_cols = [c for c in emb_df.columns if c != "subreddit"]

    # Dimensionality reduction (UMAP → 2D)
    X = emb_df[vec_cols].astype(float).values
    X_scaled = StandardScaler().fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        n_jobs=1,  
        verbose=False
    )
    coords = reducer.fit_transform(X_scaled)

    emb_df["x"], emb_df["y"] = coords[:, 0], coords[:, 1]
    emb_map = emb_df.set_index("subreddit")[["x", "y"]].to_dict("index")

    print(f"Embeddings processed: {len(emb_df)} subreddits, UMAP projection to 2D completed.")
    return emb_df, emb_map


#Prepare temporal data

def prepare_monthly_data(df: pd.DataFrame):
    """
    Adds a 'month' column (formatted as YYYY-MM) and returns the sorted list of months.
    """
    df["month"] = df["TIMESTAMP"].dt.to_period("M").astype(str)
    months = sorted(df["month"].unique())
    print(f" {len(months)} months detected ({months[0]} → {months[-1]})")
    return months



#Generate monthly Plotly frames

def create_monthly_frames(df: pd.DataFrame, months: list[str], emb_map: dict):
    """
    Creates a list of Plotly frames representing subreddit interactions
    for each month.
    """
    frames = []

    for month in tqdm(months, desc="Creating monthly frames"):
        df_month = df[df["month"] == month]

        # Aggregate links and count their frequency
        df_month_agg = (
            df_month.groupby(["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"])
            .size()
            .reset_index(name="count")
        )

        # Keep only subreddits that exist in the embeddings
        df_month_agg = df_month_agg[
            df_month_agg["SOURCE_SUBREDDIT"].isin(emb_map.keys()) &
            df_month_agg["TARGET_SUBREDDIT"].isin(emb_map.keys())
        ]

        edge_traces = []
        for _, row in df_month_agg.iterrows():
            src, tgt, sentiment, count = (
                row["SOURCE_SUBREDDIT"],
                row["TARGET_SUBREDDIT"],
                row["LINK_SENTIMENT"],
                row["count"],
            )

            x0, y0 = emb_map[src]["x"], emb_map[src]["y"]
            x1, y1 = emb_map[tgt]["x"], emb_map[tgt]["y"]

            # Red = negative, blue = positive
            color = "red" if sentiment < 0 else "blue"
            width = 0.5 + np.log1p(count)

            edge_traces.append(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(width=width, color=color),
                    opacity=0.5,
                    hoverinfo="text",
                    text=f"{src} → {tgt}<br>Sentiment={sentiment}, count={count}",
                )
            )

        # Active nodes for the month
        active_subs = set(df_month_agg["SOURCE_SUBREDDIT"]) | set(df_month_agg["TARGET_SUBREDDIT"])
        node_x, node_y, node_text = [], [], []
        for sub in active_subs:
            x, y = emb_map[sub]["x"], emb_map[sub]["y"]
            node_x.append(x)
            node_y.append(y)
            node_text.append(sub)

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=6, color="black"),
            hovertext=node_text,
            hoverinfo="text",
        )

        frames.append(
            go.Frame(
                data=edge_traces + [node_trace],
                name=month,
                layout=go.Layout(title_text=f"Emotional flows between subreddits — {month}"),
            )
        )

    return frames



# Main animation

def animate_network(df: pd.DataFrame, emb_df: pd.DataFrame):
    """
    Combines the previous functions to create an interactive Plotly animation
    showing the evolution of subreddit sentiment flows month by month.
    """

    emb_map = emb_df.set_index("subreddit")[["x", "y"]].to_dict("index")
    months = prepare_monthly_data(df)
    frames = create_monthly_frames(df, months, emb_map)

    fig = go.Figure(data=frames[0].data, frames=frames)
    fig.update_layout(
        width=950,
        height=750,
        title=f"Emotional flows between subreddits — {months[0]}",
        plot_bgcolor="white",
        showlegend=False,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 800, "redraw": True},
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                        "label": "▶️ Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                        "label": "⏸️ Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 85},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "active": 0,
                "pad": {"t": 50},
                "steps": [
                    {
                        "args": [[m], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        "label": m,
                        "method": "animate",
                    }
                    for m in months
                ],
            }
        ],
    )
    return fig
