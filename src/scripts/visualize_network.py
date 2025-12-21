import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import umap
from sklearn.preprocessing import StandardScaler

# VSCode notebook
pio.renderers.default = "vscode"


def process_embeddings_umap2d(emb_df: pd.DataFrame, random_state: int = 42):
    vec_cols = [c for c in emb_df.columns if c != "subreddit"]
    X = emb_df[vec_cols].astype(float).values
    Xs = StandardScaler().fit_transform(X)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=random_state,
        n_jobs=1,
        verbose=False,
    )
    coords = reducer.fit_transform(Xs)

    out = emb_df[["subreddit"]].copy()
    out["x"] = coords[:, 0]
    out["y"] = coords[:, 1]
    emb_map = out.set_index("subreddit")[["x", "y"]].to_dict("index")
    return out, emb_map


def animate_nodes(
    df: pd.DataFrame,
    emb_map: dict,
    min_node_links: int = 3,
    top_n_nodes: int = 1200,
    size_max: float = 22,
):
    df = df.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])
    df["month"] = df["TIMESTAMP"].dt.to_period("M").astype(str)

    months = sorted(df["month"].unique())
    emb_keys = set(emb_map.keys())

    frames = []

    for m in months:
        dm = df[df["month"] == m]
        dm = dm[
            dm["SOURCE_SUBREDDIT"].isin(emb_keys) &
            dm["TARGET_SUBREDDIT"].isin(emb_keys)
        ]

        if dm.empty:
            frames.append(go.Frame(name=m, data=[go.Scatter(x=[], y=[])]))
            continue

        out_stats = (
            dm.groupby("SOURCE_SUBREDDIT")["LINK_SENTIMENT"]
            .agg(
                out_links="size",
                mean_sent="mean",
                pos=lambda s: (s > 0).sum(),
                neg=lambda s: (s < 0).sum(),
            )
            .reset_index()
            .rename(columns={"SOURCE_SUBREDDIT": "subreddit"})
        )
        in_stats = (
            dm.groupby("TARGET_SUBREDDIT")["LINK_SENTIMENT"]
            .size()
            .reset_index(name="in_links")
            .rename(columns={"TARGET_SUBREDDIT": "subreddit"})
        )

        stats = out_stats.merge(in_stats, on="subreddit", how="left").fillna({"in_links": 0})
        stats["total_links"] = stats["out_links"] + stats["in_links"]
        stats["net_sent"] = (stats["pos"] - stats["neg"]) / stats["out_links"].clip(lower=1)

        stats = stats[stats["total_links"] >= min_node_links]
        if stats.empty:
            frames.append(go.Frame(name=m, data=[go.Scatter(x=[], y=[])]))
            continue

        stats = stats.sort_values("total_links", ascending=False).head(top_n_nodes)

        stats["x"] = stats["subreddit"].map(lambda s: emb_map[s]["x"])
        stats["y"] = stats["subreddit"].map(lambda s: emb_map[s]["y"])

        sizes = np.log1p(stats["total_links"].values)
        sizes = sizes / sizes.max() * size_max if sizes.max() > 0 else sizes
        sizes = np.clip(sizes, 4, size_max)

        trace = go.Scatter(
            x=stats["x"],
            y=stats["y"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=stats["net_sent"],
                cmin=-1, cmax=1,
                colorscale="RdBu",   # red = -1 hostile, blue = +1 positive
                opacity=0.85,
                colorbar=dict(
                    title="Net sentiment",
                    x=0.98,        
                    xanchor="left",
                    tickmode="array",
                    tickvals=[-1, 0, 1],
                    ticktext=["Hostile (-1)", "Neutral (0)", "Positive (+1)"],
                ),

                line=dict(width=0.5, color="rgba(0,0,0,0.2)"),
            ),


            text=stats["subreddit"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "total_links=%{customdata[0]}<br>"
                "out_links=%{customdata[1]}<br>"
                "in_links=%{customdata[2]}<br>"
                "mean_sent=%{customdata[3]:.3f}<br>"
                "net_sent=%{customdata[4]:.3f}<extra></extra>"
            ),
            customdata=np.stack([
                stats["total_links"].values,
                stats["out_links"].values,
                stats["in_links"].values,
                stats["mean_sent"].values,
                stats["net_sent"].values,
            ], axis=1),
            showlegend=False,
        )

        frames.append(go.Frame(name=m, data=[trace], layout=go.Layout(title_text=f"Subreddit sentiment map — {m}")))

        fig = go.Figure(data=frames[0].data, frames=frames)

    # Slider
    sliders = [{
        "active": 0,
        "pad": {"t": 30},
        "x": 0.08,
        "len": 0.88,
        "steps": [{
            "args": [[m], {"frame": {"duration": 0, "redraw": True},
                          "mode": "immediate",
                          "transition": {"duration": 0}}],
            "label": m,
            "method": "animate"
        } for m in months]
    }]

    # Play / Pause
    updatemenus = [{
        "type": "buttons",
        "direction": "left",
        "x": 0.08,
        "y": -0.05,
        "xanchor": "left",
        "yanchor": "top",
        "pad": {"r": 10, "t": 0},
        "showactive": False,
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [
                    None,
                    {
                        "frame": {"duration": 450, "redraw": True},  # speed
                        "fromcurrent": True,
                        "transition": {"duration": 0},
                        "mode": "immediate",
                    },
                ],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
            },
        ],
    }]

    fig.update_layout(
        width=None,
        height=750,
        autosize=True,
        template="plotly_white",
        title=f"Subreddit sentiment map — {months[0]}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=140, t=70, b=90),
        sliders=sliders,
        updatemenus=updatemenus,
    )


    return fig

