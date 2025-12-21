import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import mplcursors
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from IPython.display import display
from IPython.display import display
from scipy.stats import binomtest, binom
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binomtest
alt.data_transformers.disable_max_rows()


def plot_pct_multi_attacked_TARGET_SUBREDDITs(pair_month: pd.DataFrame,
                                    threshold: int = 2,
                                    month_col: str = "month",
                                    SOURCE_SUBREDDIT_col: str = "SOURCE_SUBREDDIT",
                                    TARGET_SUBREDDIT_col: str = "TARGET_SUBREDDIT",
                                    neg_col: str = "neg",
                                    ax=None) -> pd.DataFrame:
    """
    Plot the monthly percentage of TARGET_SUBREDDITs that had >= threshold distinct attackers
    (considering only negative links), and return the per-month summary.

    Returns columns:
      ['month','n_TARGET_SUBREDDITs_multi','n_TARGET_SUBREDDITs_total','pct_multi']
    """

    neg_edges = pair_month.loc[
        pair_month[neg_col] > 0,
        [month_col, SOURCE_SUBREDDIT_col, TARGET_SUBREDDIT_col]
    ].copy()

    monthly_attackers = (
        neg_edges
        .groupby([month_col, TARGET_SUBREDDIT_col])[SOURCE_SUBREDDIT_col]
        .nunique()
        .reset_index(name="n_attackers")
    )

    totals = (
        monthly_attackers
        .groupby(month_col)
        .size()
        .reset_index(name="n_TARGET_SUBREDDITs_total")
    )

    multi = (
        monthly_attackers
        .assign(is_multi=lambda df: df["n_attackers"] >= threshold)
        .groupby(month_col)["is_multi"]
        .sum()
        .reset_index(name="n_TARGET_SUBREDDITs_multi")
    )

    summary = totals.merge(multi, on=month_col, how="left").fillna(0)
    summary["pct_multi"] = (
        summary["n_TARGET_SUBREDDITs_multi"] / summary["n_TARGET_SUBREDDITs_total"]
    )

    summary = summary.sort_values(month_col)

    base = alt.Chart(summary).encode(
    x=alt.X(f"{month_col}:O", title="Month (index)", sort=list(summary[month_col])),
    y=alt.Y("pct_multi:Q", title="Percent of TARGET_SUBREDDITs (co-attacked)", axis=alt.Axis(format="%")),
    tooltip=[
        alt.Tooltip(f"{month_col}:O", title="Month"),
        alt.Tooltip("pct_multi:Q", title="Percent", format=".2%"),
        alt.Tooltip("n_TARGET_SUBREDDITs_multi:Q", title="Multi-attacked"),
        alt.Tooltip("n_TARGET_SUBREDDITs_total:Q", title="Total targets"),
    ]
        )

    line = base.mark_line(color="#df863c")

    points = base.mark_point(
    color="black",
    size=70,
    filled=True
    )

    chart = (line + points).properties(
    width=900,
    height=320,
    title=f"Share of TARGET_SUBREDDITs per Month with ≥ {threshold} Distinct Attackers"
    )

    chart.display()


    return summary


def plot_friendship_score_distribution(
    pair_monthly_scores,
    enemy_threshold,
    friend_threshold,
    bins=100
):
    """
    Plot the distribution of monthly friendship scores (log-scale histogram)
    with vertical lines indicating enemy and friend thresholds.
    """
    import pandas as pd
    import altair as alt

    scores = pair_monthly_scores["Friendship_Score"].dropna().values
    df_scores = pd.DataFrame({"Friendship_Score": scores})

    # Histogram
    hist = alt.Chart(df_scores).mark_bar().encode(
        x=alt.X(
            "Friendship_Score:Q",
            bin=alt.Bin(maxbins=bins),
            title="Monthly Friendship Score"
        ),
        y=alt.Y(
            "count():Q",
            scale=alt.Scale(type="log"),
            title="Number of (pair, month) observations (log scale)"
        ),
        tooltip=[
            alt.Tooltip("count():Q", title="Count"),
            alt.Tooltip("Friendship_Score:Q", title="Score", format=".3f")
        ]
    )

    # Threshold lines (with legend)
    thresholds_df = pd.DataFrame({
        "x": [enemy_threshold, friend_threshold],
        "Threshold": ["Enemy threshold", "Friend threshold"]
    })

    threshold_lines = alt.Chart(thresholds_df).mark_rule(
        strokeDash=[6, 4],
        strokeWidth=2
    ).encode(
        x="x:Q",
        color=alt.Color(
            "Threshold:N",
            scale=alt.Scale(
                domain=["Enemy threshold", "Friend threshold"],
                range=["red", "green"]
            ),
            legend=alt.Legend(title="Thresholds")
        ),
        tooltip=[
            alt.Tooltip("Threshold:N", title="Threshold"),
            alt.Tooltip("x:Q", title="Value", format=".3f")
        ]
    )

    chart = (hist + threshold_lines).properties(
        width=800,
        height=350,
        title="Distribution of Monthly Friendship Scores with Thresholds"
    )

    chart.display()



def plot_attack_count_distribution_from_pair_event_panel(pair_event_panel_all: pd.DataFrame,
                                                         title: str = "Distribution of Monthly Attack Counts Among Attackers of a Same Target C"):
    """
    Build the per-attacker-per-month attack counts from pair_event_panel_all
    (treating NaN as 0), then plot the distribution on a log y-scale.
    
    Returns:
      attacks_panel (DataFrame): long table with one row per (C, pair_A, pair_B, attacker, month, attacks)
      attack_counts (Series): frequency of each attack count
    """
    month_cols = [c for c in pair_event_panel_all.columns if c.startswith("month_")]

    records = []
    for row in pair_event_panel_all.itertuples(index=False):
        C, A, B = row.C, row.A, row.B

        for mcol in month_cols:
            val = getattr(row, mcol)
            month_idx = int(mcol.split("_")[1])

            if isinstance(val, tuple):
                attacks_A, attacks_B = val
            else:
                attacks_A = 0
                attacks_B = 0

            records.append({
                "C": C,
                "pair_A": A,
                "pair_B": B,
                "attacker": A,
                "month": month_idx,
                "attacks": attacks_A,
            })

            records.append({
                "C": C,
                "pair_A": A,
                "pair_B": B,
                "attacker": B,
                "month": month_idx,
                "attacks": attacks_B,
            })

    attacks_panel = pd.DataFrame(records)

    attack_counts = attacks_panel["attacks"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    plt.bar(attack_counts.index, attack_counts.values, edgecolor="black")

    plt.xlabel("Number of attacks per attacker per month")
    plt.ylabel("Number of observations (log scale)")
    plt.title(title)

    plt.yscale("log")
    plt.xticks(attack_counts.index)
    plt.tight_layout()
    plt.show()

    return attacks_panel, attack_counts


def plot_conflict_friendship_timelines_basic(pair_summary_strict,
                                             friendship_stat_strict,
                                             idx_to_month,
                                             C_SEL="leagueoflegends",
                                             TOPN=25):
    """
    Plot conflict_start and friendship_start for pairs (A,B) that co-attacked C_SEL,
    using only pair_summary_strict and friendship_stat_strict.

    - pair_summary_strict: has columns ['C','A','B','start_month','active_months', 'end_month' (optional)]
      where start_month is the first strong co-attack month.
    - friendship_stat_strict: has columns ['A','B','start_month','active_months']
      where start_month is first strict friendship month.
    - idx_to_month: maps month index -> real month string.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Rename for clarity before merging
    ps = pair_summary_strict.rename(
        columns={"start_month": "conflict_start",
                 "active_months": "conflict_active_months"}
    )
    fs = friendship_stat_strict.rename(
        columns={"start_month": "friendship_start",
                 "active_months": "friendship_active_months"}
    )

    # Merge on (A,B); keep C from pair_summary_strict
    df = ps.merge(fs, on=["A", "B"], how="left")

    # Filter to desired target C
    dfC = df.loc[df["C"] == C_SEL].copy()
    if dfC.empty:
        raise ValueError(f"No co-attacker pairs found for target C = {C_SEL!r}")

    # Keep only pairs where we actually have a friendship_start (>0)
    dfC = dfC.loc[dfC["friendship_start"] > 0].copy()
    if dfC.empty:
        raise ValueError(
            f"No pairs with observed strict friendship for target C = {C_SEL!r}"
        )

    # Compute lag (can be negative if friendship pre-dates conflict)
    dfC["delta"] = dfC["friendship_start"] - dfC["conflict_start"]

    # Sort by lag and take top N
    dfC = dfC.sort_values("delta").head(TOPN)

    # Map numeric month indices to real month labels
    dfC["conflict_month"] = dfC["conflict_start"].map(idx_to_month)
    dfC["friendship_month"] = dfC["friendship_start"].map(idx_to_month)

    # --- Plot ---
        # --- Plot ---
    import altair as alt

    df_plot = dfC.reset_index(drop=True).copy()
    df_plot["pair"] = df_plot["A"].astype(str) + " ↔ " + df_plot["B"].astype(str)

    rule = alt.Chart(df_plot).mark_rule(color="#4a4a4a", strokeWidth=2).encode(
        x=alt.X("conflict_start:Q", title="Month"),
        x2="friendship_start:Q",
        y=alt.Y("pair:O", title="Attacker Pair (A,B)", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    points_conflict = alt.Chart(df_plot).mark_point(filled=True, size=80, color="red").encode(
        x="conflict_start:Q",
        y=alt.Y("pair:O", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    points_friendship = alt.Chart(df_plot).mark_point(filled=True, size=80, color="green").encode(
        x="friendship_start:Q",
        y=alt.Y("pair:O", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    chart = (rule + points_conflict + points_friendship).properties(
        width=900,
        height=max(120, 22 * len(df_plot) + 40),
        title=f"Conflict vs Friendship Start — Target C = {C_SEL}"
    )

    from IPython.display import display
    display(chart)


    return dfC


def plot_friendship_outcomes_pie(comparison):
    import matplotlib.pyplot as plt
    import numpy as np
    import plotly.graph_objects as go
    import plotly.io as pio

    # --- masks / counts ---
    new_mask    = comparison['new_friendship']
    far_mask    = comparison['far_friendship']
    never_mask  = ~comparison['friendship_observed']
    stayed_mask = comparison['friendship_observed'] & ~comparison['friendship_after']

    n = len(comparison)
    counts = [
        new_mask.sum(),
        far_mask.sum(),
        never_mask.sum(),
        stayed_mask.sum()
    ]

    labels_defs = [
        "New friends (soon after conflict)",
        "Far friends (long after conflict)",
        "Never friends (no friendship)",
        "Were and Remained friends"
    ]

    sizes = [c / n for c in counts]

    # --- plot ---
    import altair as alt
    from IPython.display import display
    import pandas as pd

    df_plot = pd.DataFrame({
        "Outcome": labels_defs,
        "Count": counts,
        "Share": sizes
    })

    chart = alt.Chart(df_plot).mark_arc().encode(
        theta=alt.Theta("Share:Q"),
        color=alt.Color("Outcome:N", legend=alt.Legend(
            title="Outcome (count, % of pairs)"
        )),
        tooltip=[
            alt.Tooltip("Outcome:N", title="Outcome"),
            alt.Tooltip("Count:Q", title="Count"),
            alt.Tooltip("Share:Q", title="Share", format=".2%")
        ]
    ).properties(
        width=450,
        height=350,
        title="Co-attacking Pair Friendship Outcomes"
    )
    display(chart)


def plot_new_friendship_timelines(comparison, idx_to_month, C_SEL="leagueoflegends", TOPN=25):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    # --- Step 1: Filter to pairs that became new friends ---
    df = comparison.loc[comparison["new_friendship"] == True].copy()

    # Auto-pick the target with most new friendships if not specified
    if C_SEL is None:
        C_SEL = df["C"].value_counts().idxmax()

    dfC = df.loc[df["C"] == C_SEL].copy()
    if dfC.empty:
        raise ValueError(f"No new_friendship rows found for target C = {C_SEL!r}")

    # Compute time lag
    dfC["delta"] = dfC["friendship_start"] - dfC["conflict_start"]
    dfC = dfC.loc[dfC["delta"] > 0].sort_values("delta").head(TOPN)

    # Map indices to readable months
    dfC["conflict_month"] = dfC["conflict_start"].map(idx_to_month)
    dfC["friendship_month"] = dfC["friendship_start"].map(idx_to_month)
    dfC["conflict_end_month"] = dfC["conflict_end"].map(idx_to_month)      # <<< ADDED

    # --- Step 3: Plot timelines ---
    plt.figure(figsize=(10, 0.4 * len(dfC) + 2))

    import altair as alt
    from IPython.display import display

    df_plot = dfC.reset_index(drop=True).copy()
    df_plot["pair"] = df_plot["A"].astype(str) + " ↔ " + df_plot["B"].astype(str)

    rule = alt.Chart(df_plot).mark_rule(color="#4a4a4a", strokeWidth=2).encode(
        x=alt.X("conflict_start:Q", title="Month"),
        x2="friendship_start:Q",
        y=alt.Y("pair:O", title="Attacker Pair (A,B)", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("conflict_end:Q", title="Conflict end (idx)"),
            alt.Tooltip("conflict_end_month:O", title="Conflict end"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    points_conflict = alt.Chart(df_plot).mark_point(filled=True, size=80, color="red").encode(
        x="conflict_start:Q",
        y=alt.Y("pair:O", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("conflict_end:Q", title="Conflict end (idx)"),
            alt.Tooltip("conflict_end_month:O", title="Conflict end"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    points_conflict_end = alt.Chart(df_plot).mark_point(filled=True, size=80, color="orange").encode(
        x="conflict_end:Q",
        y=alt.Y("pair:O", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("conflict_end:Q", title="Conflict end (idx)"),
            alt.Tooltip("conflict_end_month:O", title="Conflict end"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    points_friendship = alt.Chart(df_plot).mark_point(filled=True, size=80, color="green").encode(
        x="friendship_start:Q",
        y=alt.Y("pair:O", sort=list(df_plot["pair"])),
        tooltip=[
            alt.Tooltip("pair:O", title="Pair"),
            alt.Tooltip("conflict_start:Q", title="Conflict start (idx)"),
            alt.Tooltip("conflict_month:O", title="Conflict start"),
            alt.Tooltip("conflict_end:Q", title="Conflict end (idx)"),
            alt.Tooltip("conflict_end_month:O", title="Conflict end"),
            alt.Tooltip("friendship_start:Q", title="Friendship start (idx)"),
            alt.Tooltip("friendship_month:O", title="Friendship start"),
            alt.Tooltip("delta:Q", title="Lag (months)")
        ]
    )

    chart = (rule + points_conflict + points_conflict_end + points_friendship).properties(
        width=900,
        height=max(120, 22 * len(df_plot) + 40),
        title=f"When Co-Attackers Became Friends — Target C = {C_SEL}"
    )

    display(chart)

    return dfC


import networkx as nx

def plot_causal_dag():
    """
    Plot the causal DAG relating confounders, co-attack (treatment),
    and friendship (outcome).
    """

    # Create directed graph
    G = nx.DiGraph()

    # Nodes
    nodes = [
        "Similarity", "Aggressiveness", "Activity", "Hostility",
        "CoAttack", "Friendship"
    ]
    G.add_nodes_from(nodes)

    # Edges: confounders → treatment and outcome
    confounders = ["Similarity", "Aggressiveness", "Activity"]
    for c in confounders:
        G.add_edge(c, "CoAttack")
        G.add_edge(c, "Friendship")

    # Hostility → Friendship only
    G.add_edge("Hostility", "Friendship")

    # Treatment → Outcome
    G.add_edge("CoAttack", "Friendship")

    # Layout
    pos = {
        "Similarity":     (-1.8,  1.0),
        "Aggressiveness": (-0.6,  1.6),
        "Activity":       (0.6,   1.0),
        "Hostility":      (1.8,   1.6),
        "CoAttack":       (0.0,   0.0),
        "Friendship":     (0.0,  -1.2),
    }

    # Node colors
    node_colors = {
        "Similarity":     "#fde68a",
        "Aggressiveness": "#fca5a5",
        "Activity":       "#93c5fd",
        "Hostility":      "#fdba74",
        "CoAttack":       "#a7f3d0",
        "Friendship":     "#c4b5fd",
    }

    plt.figure(figsize=(8, 6))

    # Draw nodes
    for node in nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[node],
            node_color=node_colors[node],
            node_size=1300
        )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

    # Draw edges with arrowheads
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=1,
        edge_color="black",
        connectionstyle="arc3,rad=0.0",
        min_source_margin=15,
        min_target_margin=15
    )

    plt.axis("off")
    plt.title("Causal DAG: Co-Attack, Friendship, and Confounders", fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_logit_coefficients(logit, confounders):
    """
    Bar plot of logistic regression coefficients (on standardized confounders).
    Larger |coef| => stronger association with being treated.
    Sign: + increases treatment probability, - decreases it.
    """
    coefs = logit.coef_.ravel()
    order = np.argsort(np.abs(coefs))[::-1]
    conf_sorted = [confounders[i] for i in order]
    coefs_sorted = coefs[order]

    import pandas as pd
    import altair as alt
    from IPython.display import display

    df_plot = pd.DataFrame({
        "Confounder": conf_sorted,
        "Coefficient": coefs_sorted
    })

    chart = alt.Chart(df_plot).mark_bar(
        color="#f4a261"
    ).encode(
        x=alt.X(
            "Confounder:N",
            sort=None,
            title=None
        ),
        y=alt.Y(
            "Coefficient:Q",
            title="Log-odds coefficient"
        ),
        tooltip=[
            alt.Tooltip("Confounder:N", title="Confounder"),
            alt.Tooltip("Coefficient:Q", title="Coefficient", format=".3f")
        ]
    ).properties(
        width=700,
        height=350,
        title="Propensity Model Coefficients (Standardized Confounders)"
    )

    zero_line = alt.Chart(
        pd.DataFrame({"y": [0]})
    ).mark_rule(
        color="black",
        strokeWidth=1
    ).encode(
        y="y:Q"
    )

    display(chart + zero_line)


# Plot B — ROC curve + AUC (how well confounders predict treatment)
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc_for_propensity(df_ps):
    """
    ROC curve of predicting treated from confounders (via pscore).
    AUC close to 0.5 => weak separation; close to 1 => strong separation.
    """
    y_true = df_ps["treated"].astype(int).values
    y_score = df_ps["pscore"].values

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    import pandas as pd
    import altair as alt
    from IPython.display import display

    df_roc = pd.DataFrame({
        "FPR": fpr,
        "TPR": tpr,
        "Curve": "Propensity model"
    })

    df_diag = pd.DataFrame({
        "FPR": [0, 1],
        "TPR": [0, 1],
        "Curve": "Random (AUC = 0.5)"
    })

    df_plot = pd.concat([df_roc, df_diag], ignore_index=True)

    chart = alt.Chart(df_plot).mark_line().encode(
        x=alt.X("FPR:Q", title="False Positive Rate"),
        y=alt.Y("TPR:Q", title="True Positive Rate"),
        color=alt.Color(
            "Curve:N",
            scale=alt.Scale(
                domain=["Propensity model", "Random (AUC = 0.5)"],
                range=["black", "#f4a261"]
            ),
            legend=alt.Legend(title=None)
        ),
        strokeDash=alt.StrokeDash(
            "Curve:N",
            scale=alt.Scale(
                domain=["Propensity model", "Random (AUC = 0.5)"],
                range=[[1, 0], [6, 4]]
            )
        ),
        tooltip=[
            alt.Tooltip("Curve:N", title="Curve"),
            alt.Tooltip("FPR:Q", title="False Positive Rate", format=".3f"),
            alt.Tooltip("TPR:Q", title="True Positive Rate", format=".3f")
        ]
    ).properties(
        width=450,
        height=450,
        title=f"ROC Curve — Propensity Model (AUC = {auc:.3f})"
    )

    display(chart)


   # ------------------------------------------------------------
# 4.4 Plot confounder distributions BEFORE matching
# ------------------------------------------------------------

def plot_confounders_separately(df_mix, confounders):
    """
    Plot one boxplot per confounder (linear scale),
    comparing treated vs control groups.
    """
    df_plot = df_mix.copy()
    df_plot["group"] = df_plot["treated"].map({1: "Treated", 0: "Control"})

    for conf in confounders:
        plt.figure(figsize=(6, 4))
        sns.boxplot(
            data=df_plot,
            x="group",
            y=conf
        )
        plt.title(f"{conf} — Treated vs Control (Before Matching)")
        plt.xlabel("")
        plt.ylabel(conf)
        plt.tight_layout()
        plt.show()


# ------------------------------------------------------------
# 4.5 Plot diagnostics: overlap of propensity scores
# ------------------------------------------------------------
def plot_pscore_overlap(df):
    plt.figure(figsize=(10,6))
    sns.kdeplot(data=df[df.treated == 1], x="pscore", fill=True, alpha=0.5, label="Treated")
    sns.kdeplot(data=df[df.treated == 0], x="pscore", fill=True, alpha=0.5, label="Control")

    plt.title("Propensity Score Distributions — Treated vs Control")
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

def plot_confounders_separately_after_matching(matched_df, confounders):
    """
    One boxplot per confounder (linear scale), Treated vs Control,
    using the matched dataset.
    """
    df_plot = matched_df.copy()
    df_plot["group"] = df_plot["treated"].map({1: "Treated", 0: "Control"})

    for conf in confounders:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_plot, x="group", y=conf)
        plt.title(f"{conf} — Treated vs Control (After Matching)")
        plt.xlabel("")
        plt.ylabel(conf)
        plt.tight_layout()
        plt.show()

def plot_pscore_distribution_after_matching(matched_df):
    """
    Plot propensity score distributions for treated vs control
    after matching.
    """
    plt.figure(figsize=(8, 5))

    sns.kdeplot(
        data=matched_df[matched_df.treated == 1],
        x="pscore", fill=True, alpha=0.5, label="Treated"
    )
    sns.kdeplot(
        data=matched_df[matched_df.treated == 0],
        x="pscore", fill=True, alpha=0.5, label="Control"
    )

    plt.title("Propensity Score Distribution After Matching")
    plt.xlabel("Propensity score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Plot 1: Match quality (distance distribution)
# ------------------------------------------------------------
def plot_match_distance(matched_df):
    """
    Histogram of nearest-neighbor distances (|pscore_t - pscore_c|)
    for the treated rows (one per matched pair).
    Smaller distances = better matches.
    """
    treated_rows = matched_df[matched_df.treated == 1].copy()

    import pandas as pd
    import altair as alt
    from IPython.display import display

    df_plot = treated_rows[["match_dist"]].copy()

    hist = alt.Chart(df_plot).mark_bar(
        color="#e97a20",
        opacity=0.7
    ).encode(
        x=alt.X(
            "match_dist:Q",
            bin=alt.Bin(maxbins=40),
            title="|pscore_treated - pscore_control|"
        ),
        y=alt.Y(
            "count():Q",
            title="Count"
        ),
        tooltip=[
            alt.Tooltip("count():Q", title="Count")
        ]
    )

    kde = alt.Chart(df_plot).transform_density(
        "match_dist",
        as_=["match_dist", "density"]
    ).mark_line(
        color="black",
        strokeWidth=2
    ).encode(
        x="match_dist:Q",
        y=alt.Y("density:Q", title="Density")
    )

    chart = hist.properties(
    width=600,
    height=350,
    title="Match Quality: Distribution of NN Distances (Propensity Score)"
)

    display(chart)


# ------------------------------------------------------------
# Plot 2: pscore scatter by matched pair (treated vs control)
# ------------------------------------------------------------
def plot_pscore_pairs(matched_df, max_pairs=200):
    """
    Visualize treated vs control propensity scores per matched_pair_id.
    If many pairs, subsample for readability.
    """
    df = matched_df[["matched_pair_id", "treated", "pscore"]].copy()
    n_pairs = df["matched_pair_id"].nunique()

    if n_pairs > max_pairs:
        keep_ids = np.random.default_rng(0).choice(df["matched_pair_id"].unique(), size=max_pairs, replace=False)
        df = df[df["matched_pair_id"].isin(keep_ids)].copy()

    import altair as alt
    from IPython.display import display
    import pandas as pd

    df_plot = df.copy()
    df_plot["treated_label"] = df_plot["treated"].map({0: "Control (0)", 1: "Treated (1)"})

    chart = alt.Chart(df_plot).mark_circle(
        size=35,
        opacity=0.7
    ).encode(
        x=alt.X(
            "matched_pair_id:N",
            title="matched_pair_id",
            axis=alt.Axis(labels=False, ticks=False)
        ),
        y=alt.Y(
            "pscore:Q",
            title="pscore"
        ),
        color=alt.Color(
            "treated_label:N",
            scale=alt.Scale(
                domain=["Control (0)", "Treated (1)"],
                range=["#2d6be8", "#f2730b"]
            ),
            legend=alt.Legend(title="treated")
        ),
        xOffset=alt.XOffset("treated_label:N"),
        tooltip=[
            alt.Tooltip("matched_pair_id:N", title="matched_pair_id"),
            alt.Tooltip("treated_label:N", title="treated"),
            alt.Tooltip("pscore:Q", title="pscore", format=".3f")
        ]
    ).properties(
        width=750,
        height=320,
        title="Propensity Scores Within Matched Pairs (subsampled)"
    )

    display(chart)


def plot_treatment_outcome_matrix(df, normalize=False):
    """
    2x2 matrix of Treatment (rows) vs Outcome Y (cols).
    If normalize=True, shows row-wise proportions instead of counts.
    """
    import altair as alt
    from IPython.display import display
    import pandas as pd
    
    tab = pd.crosstab(
        df["treated"], df["Y"],
        normalize="index" if normalize else False
    )

    tab.index = tab.index.map({0: "Control", 1: "Treated"})
    tab.columns = tab.columns.map({0: "No friendship (Y=0)", 1: "Friendship (Y=1)"})


    df_plot = tab.reset_index().melt(
        id_vars="treated",
        var_name="Outcome",
        value_name="Value"
    )

    chart = alt.Chart(df_plot).mark_rect(
    stroke="black",
    strokeWidth=1
    ).encode(
        x=alt.X(
            "Outcome:N",
            title="Outcome"
        ),
        y=alt.Y(
            "treated:N",
            title=None
        ),
        color=alt.Color(
        "Value:Q",
        scale=alt.Scale(
            domain=[0, 1] if normalize else [0, df_plot["Value"].quantile(0.90)],
            range=["#ffd29f", "#faab57", "#fd7931"]
        ),
        legend=None
        ),
        tooltip=[
            alt.Tooltip("treated:N", title="Treatment"),
            alt.Tooltip("Outcome:N", title="Outcome"),
            alt.Tooltip(
                "Value:Q",
                title="Proportion" if normalize else "Count",
                format=".3f" if normalize else "d"
            )
        ]
    ).properties(
        width=420,
        height=220,
        title="Treatment vs Outcome"
        + (" (Row-wise proportions)" if normalize else " (Counts)")
    )

    text = alt.Chart(df_plot).mark_text(
        color="black",
        fontSize=12
    ).encode(
        x="Outcome:N",
        y="treated:N",
        text=alt.Text(
            "Value:Q",
            format=".3f" if normalize else "d"
        )
    )

    display(chart + text)


def plot_bootstrap_att(results, ci=95):
    """
    Visualize bootstrap distribution of ATT with CI and null reference.
    """
    boot = results["boot_means"]
    att  = results["att"]
    lo   = results["ci_low"]
    hi   = results["ci_high"]

    import pandas as pd
    import altair as alt
    from IPython.display import display

    df_plot = pd.DataFrame({"ATT_pp": boot * 100})

    # Histogram
    hist = alt.Chart(df_plot).mark_bar(
        color="lightgray",
        opacity=0.7
    ).encode(
        x=alt.X(
            "ATT_pp:Q",
            bin=alt.Bin(maxbins=40),
            title="ATT (percentage points)",
            axis=alt.Axis(tickCount=10, format=".1f")
        ),
        y=alt.Y(
            "count():Q",
            title=""
        ),
        tooltip=[
            alt.Tooltip("count():Q", title="Count")
        ]
    )

    # KDE
    kde = alt.Chart(df_plot).transform_density(
        "ATT_pp",
        as_=["ATT_pp", "density"]
    ).mark_line(
        strokeWidth=2
    ).encode(
        x="ATT_pp:Q",
        y=alt.Y("density:Q", title="Density"),
        color=alt.value("black")
    )

    # Reference lines + CI (with legend)
    ref_df = pd.DataFrame({
        "x": [0.0, att * 100],
        "label": ["Null (ATT = 0)", "Observed ATT"]
    })

    ref_lines = alt.Chart(ref_df).mark_rule(
        strokeWidth=2
    ).encode(
        x="x:Q",
        color=alt.Color(
            "label:N",
            scale=alt.Scale(
                domain=["Null (ATT = 0)", "Observed ATT"],
                range=["black", "red"]
            ),
            legend=alt.Legend(title="Reference")
        ),
        strokeDash=alt.StrokeDash(
            "label:N",
            scale=alt.Scale(
                domain=["Null (ATT = 0)", "Observed ATT"],
                range=[[6, 4], [1, 0]]
            )
        ),
        tooltip=[
            alt.Tooltip("label:N", title="Reference"),
            alt.Tooltip("x:Q", title="ATT (pp)", format=".2f")
        ]
    )

    # CI band
    ci_band = alt.Chart(
        pd.DataFrame({
            "x1": [lo * 100],
            "x2": [hi * 100],
            "label": [f"{ci}% CI"]
        })
    ).mark_rect(
        opacity=0.2,
        color="red"
    ).encode(
        x="x1:Q",
        x2="x2:Q",
        tooltip=[
            alt.Tooltip("label:N", title="Interval"),
            alt.Tooltip("x1:Q", title="CI low (pp)", format=".2f"),
            alt.Tooltip("x2:Q", title="CI high (pp)", format=".2f")
        ]
    )

    chart = (hist + ci_band + ref_lines).properties(
        width=650,
        height=380,
        title="Bootstrap Distribution of ATT"
    )

    display(chart)


def plot_sensitivity_curve(
    bounds_df: pd.DataFrame,
    gamma_star: float | None = None,
    alpha: float = 0.05,
    ax=None,
    show: bool = True,
    interactive: bool = True,
):
    """
    Rosenbaum sensitivity curve: worst-case p-value vs Γ.
    """
    if bounds_df is None or bounds_df.empty:
        print("No bounds to plot (bounds_df is empty).")
        return None

    df = bounds_df.copy().sort_values("Gamma")

    import altair as alt
    from IPython.display import display
    import pandas as pd

    # ---- Base chart (x starts at 0) ----
    base = alt.Chart(df).encode(
        x=alt.X(
            "Gamma:Q",
            title="Hidden bias parameter Γ",
            scale=alt.Scale(domainMin=0)
        )
    )

    # ---- Sensitivity curve (blue line) ----
    line = base.mark_line(
        color="steelblue",
        strokeWidth=2
    ).encode(
        y=alt.Y("p_upper:Q", title="Worst-case p-value"),
        tooltip=[
            alt.Tooltip("Gamma:Q", title="Γ", format=".2f"),
            alt.Tooltip("p_upper:Q", title="Worst-case p-value", format=".4f"),
        ]
    )

    # ---- Points (black dots) ----
    points = base.mark_point(
        color="black",
        filled=True,
        size=60
    ).encode(
        y="p_upper:Q",
        tooltip=[
            alt.Tooltip("Gamma:Q", title="Γ", format=".2f"),
            alt.Tooltip("p_upper:Q", title="Worst-case p-value", format=".4f"),
        ]
    )

    # ---- Reference lines data (for legend) ----
    ref_rows = [{"value": alpha, "axis": "y", "Reference": f"α = {alpha}"}]

    if gamma_star is not None:
        ref_rows.append(
            {"value": gamma_star, "axis": "x", "Reference": rf"Γ* ≈ {gamma_star}"}
        )

    ref_df = pd.DataFrame(ref_rows)

    # ---- Horizontal α line ----
    alpha_line = alt.Chart(
        ref_df[ref_df["axis"] == "y"]
    ).mark_rule(
        strokeWidth=2,
        strokeDash=[6, 4]
    ).encode(
        y="value:Q",
        color=alt.Color(
            "Reference:N",
            legend=alt.Legend(title="Reference")
        )
    )

    layers = line + points + alpha_line

    # ---- Vertical Γ* line ----
    if gamma_star is not None:
        gamma_line = alt.Chart(
            ref_df[ref_df["axis"] == "x"]
        ).mark_rule(
            strokeWidth=2,
            strokeDash=[2, 2]
        ).encode(
            x="value:Q",
            color=alt.Color(
                "Reference:N",
                legend=alt.Legend(title="Reference")
            )
        )
        layers = layers + gamma_line

    chart = layers.properties(
        width=650,
        height=380,
        title="Rosenbaum Sensitivity Analysis (Sign Test)"
    )

    display(chart)
    return chart

