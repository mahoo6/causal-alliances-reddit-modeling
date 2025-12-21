import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "vscode"


def load_hidden_hostility_monthly_from_df(df_hostility: pd.DataFrame) -> pd.DataFrame:
    """
    Expects df_hostility columns:
    SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP, LINK_SENTIMENT, ...
    s_flip, is_hidden_hostility

    Returns a df with: month, subreddit, hostility_emitted
    where hostility_emitted = # outgoing links with is_hidden_hostility==1
    per (month, SOURCE_SUBREDDIT).
    """
    h = df_hostility.copy()

    required = {"TIMESTAMP", "SOURCE_SUBREDDIT", "is_hidden_hostility"}
    missing = required - set(h.columns)
    if missing:
        raise ValueError(f"df_hostility missing columns: {missing}")

    h["TIMESTAMP"] = pd.to_datetime(h["TIMESTAMP"], errors="coerce")
    h = h.dropna(subset=["TIMESTAMP"])
    h["month"] = h["TIMESTAMP"].dt.to_period("M").astype(str)

    hh = h[h["is_hidden_hostility"].astype(int) == 1]

    hm = (
        hh.groupby(["month", "SOURCE_SUBREDDIT"])
          .size()
          .reset_index(name="hostility_emitted")
          .rename(columns={"SOURCE_SUBREDDIT": "subreddit"})
    )
    return hm


def animate_hidden_hostility_nodes(
    df_links: pd.DataFrame,
    emb_map: dict,
    df_hostility: pd.DataFrame,
    min_node_links: int = 3,
    top_n_nodes: int = 1200,
    size_max: float = 26,
):
    # --- df_links prep
    df = df_links.copy()
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    df = df.dropna(subset=["TIMESTAMP"])
    df["month"] = df["TIMESTAMP"].dt.to_period("M").astype(str)

    months = sorted(df["month"].unique())
    if len(months) == 0:
        raise ValueError("No valid months found in df_links TIMESTAMP column.")

    emb_keys = set(emb_map.keys())

    # --- hostility monthly (by source subreddit)
    hm = load_hidden_hostility_monthly_from_df(df_hostility)
    hm = hm[hm["subreddit"].isin(emb_keys)].copy()

    frames = []

    for m in months:
        dm = df[df["month"] == m]
        dm = dm[
            dm["SOURCE_SUBREDDIT"].isin(emb_keys)
            & dm["TARGET_SUBREDDIT"].isin(emb_keys)
        ]

        if dm.empty:
            frames.append(go.Frame(name=m, data=[go.Scatter(x=[], y=[])]))
            # continue to next month
            continue


        # --- context stats from df_links
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

        # filter: only show sufficiently active nodes this month
        stats = stats[stats["total_links"] >= min_node_links]
        if stats.empty:
            frames.append(go.Frame(name=m, data=[go.Scatter(x=[], y=[])]))
            continue

        # keep top connected nodes
        stats = stats.sort_values("total_links", ascending=False).head(top_n_nodes).copy()

        # merge hostility emitted for this month (0 if none)
        hm_m = hm[hm["month"] == m][["subreddit", "hostility_emitted"]]
        stats = stats.merge(hm_m, on="subreddit", how="left")
        stats["hostility_emitted"] = stats["hostility_emitted"].fillna(0.0)

        # coords
        stats["x"] = stats["subreddit"].map(lambda s: emb_map[s]["x"])
        stats["y"] = stats["subreddit"].map(lambda s: emb_map[s]["y"])

        # color: red if emitted this month, else grey
        stats["color"] = np.where(
            stats["hostility_emitted"] > 0,
            "rgb(220,0,0)",
            "rgb(170,170,170)",
        )

        # size: log-scaled within-month normalization
        raw = stats["hostility_emitted"].values.astype(float)
        sizes = np.log1p(raw)
        if sizes.max() > 0:
            sizes = sizes / sizes.max() * size_max
            sizes = np.clip(sizes, 4, size_max)
        else:
            sizes = np.full(len(stats), 6.0)

        trace = go.Scatter(
            x=stats["x"],
            y=stats["y"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=stats["color"],
                opacity=0.85,
                line=dict(width=0.5, color="rgba(0,0,0,0.25)"),
            ),
            text=stats["subreddit"],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "hidden_hostility_emitted=%{customdata[0]}<br>"
                "total_links=%{customdata[1]}<br>"
                "out_links=%{customdata[2]}<br>"
                "in_links=%{customdata[3]}<br>"
                "mean_sent=%{customdata[4]:.3f}<br>"
                "net_sent=%{customdata[5]:.3f}<extra></extra>"
            ),
            customdata=np.stack(
                [
                    stats["hostility_emitted"].values,
                    stats["total_links"].values,
                    stats["out_links"].values,
                    stats["in_links"].values,
                    stats["mean_sent"].values,
                    stats["net_sent"].values,
                ],
                axis=1,
            ),
            showlegend=False,
        )

        frames.append(
            go.Frame(
                name=m,
                data=[trace],
                layout=go.Layout(title_text=f"Hidden-hostility emitters — {m}"),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)

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
                "args": [None, {
                    "frame": {"duration": 450, "redraw": True},
                    "fromcurrent": True,
                    "transition": {"duration": 0},
                    "mode": "immediate",
                }],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
            },
        ],
    }]

    fig.update_layout(
        width=None,
        height=750,
        autosize=True,
        template="plotly_white",
        title=f"Hidden-hostility emitters — {months[0]}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=40, t=70, b=90),
        sliders=sliders,
        updatemenus=updatemenus,
    )

    return fig
