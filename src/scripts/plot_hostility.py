import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from matplotlib import cm

# ============================
# General Analysis
# ============================
def plot_general_analysis(df: pd.DataFrame):
    (
        (df['LINK_SENTIMENT'] < 0)
        .value_counts(normalize=True)
        .rename(index={False: 'non-negative', True: 'negative'})
        .reindex(['non-negative', 'negative'], fill_value=0)
        .plot.bar()
    )

# ============================
# Embedding Analysis
# ============================
def plot_embedding_kde_and_test(df_hostility: pd.DataFrame):
    plot_df = df_hostility.dropna(subset=['cosine_sim']).copy()

    plt.figure()
    sns.kdeplot(
        data=plot_df,
        x='cosine_sim',
        hue='is_neg',
        common_norm=False,
        fill=True,
        hue_order=[0, 1],
        palette=['#aaaaaa', '#cc3333'],
    )
    plt.xlabel('Cosine similarity (source vs target embedding)')
    plt.ylabel('Density')
    plt.title('Similarity of communities by link sentiment')
    plt.legend(title='is_neg', labels=['non-negative','negative'])
    plt.show()

    # Compare cosine similarity distributions
    neg = df_hostility.loc[df_hostility.is_neg == 1, 'cosine_sim'].dropna()
    nonneg = df_hostility.loc[df_hostility.is_neg == 0, 'cosine_sim'].dropna()
    u, pval = mannwhitneyu(neg, nonneg, alternative='two-sided')
    print("Mann–Whitney p-value:", pval)

# ============================
# time-to-flip plots
# ============================

PALETTE = list(cm.get_cmap("tab20").colors)

BAR_COLORS = ("#1f77b4", "#aec7e8")  # two blues for the bar chart
HIST_COLOR = "#ff7f0e" 


def plot_followups_pairs_grid(
    followups_dict: dict[str, pd.DataFrame],
    hours_map: dict[str, float] | None = None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    if hours_map is None:
        hours_map = {}

    labels = list(followups_dict.keys())
    n = len(labels)

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), constrained_layout=True)
    if n == 1:
        axes = np.array([axes])  # normalize to shape (n, 2)

    for i, label in enumerate(labels):
        f = followups_dict[label]

        # Window label for titles (fallback to dict key)
        try:
            win_label = str(f["Window_Label"].iloc[0])
        except Exception:
            win_label = label

        # Prep data
        df = f.copy()
        df["delay_hours"] = df["Time_Difference"].dt.total_seconds() / 3600.0
        df["flip_kind"] = np.where(df["Link_Sentiment1"] == 1, "+1 \u2192 -1", "-1 \u2192 +1")

        # ----- Left: bar (flip counts)
        ax1 = axes[i, 0]
        counts = df["flip_kind"].value_counts().reindex(["+1 \u2192 -1", "-1 \u2192 +1"]).fillna(0)
        ax1.bar(counts.index, counts.values, color=list(BAR_COLORS))
        ax1.set_ylabel("Count")
        ax1.set_xlabel("Flip direction")
        ax1.set_title(f"Opposite follow-ups within {win_label}")

        # ----- Right: histogram (delay distribution)
        ax2 = axes[i, 1]
        max_hours = hours_map.get(label)
        if max_hours is None:
            max_hours = float(np.nanmax(df["delay_hours"])) if len(df) else 24.0
        bins = np.linspace(0, max_hours, 51)
        ax2.hist(df["delay_hours"].dropna(), bins=bins, align="left", color=HIST_COLOR)
        ax2.set_xlabel("Delay between opposite links (hours)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Delay distribution ({label})")

    plt.show()


def plot_VADER(df):
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="LINK_SENTIMENT", y="VADER_compound")
    plt.xticks([0,1], ["-1 (négatif)", "+1 (non-négatif)"])
    plt.title("VADER compound par label")
    plt.show()