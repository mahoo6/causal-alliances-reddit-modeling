import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
from scipy.stats import pointbiserialr

def summarize_memory(df: pd.DataFrame):
    """Displays the memory usage of the DataFrame."""
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"Memory usage: {mem_mb:.2f} MB")
    return mem_mb

def describe_features(df: pd.DataFrame, property_names):
    """Displays descriptive statistics for the selected features."""
    desc = df[property_names].describe().T
    print(desc.head(10))
    return desc

def plot_links_per_month(df: pd.DataFrame):
    """Plots the number of links per month."""
    df.set_index("TIMESTAMP").resample("M").size().plot()
    plt.title("Links per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of Links")
    plt.show()

def point_biserial_analysis(df: pd.DataFrame, property_names):
    """Performs point-biserial correlation analysis between each feature and the sentiment label,
    and plots the most correlated features."""
    df["label_bin"] = df["LINK_SENTIMENT"].replace({-1: 0, 1: 1})
    results = []
    for col in property_names:
        if col in df.columns:
            r, p = pointbiserialr(df["label_bin"], df[col])
            results.append({"feature": col, "r": r, "p_value": p})
    corr_df = pd.DataFrame(results).sort_values("p_value")

    print("=== Top 10 features most correlated with hostility (Point-Biserial) ===")
    print(corr_df.head(10))
    print("\n=== 10 strongest negative correlations (more hostile features) ===")
    print(corr_df.sort_values("r").head(10))

    # --- Plotting section ---
    top_n = 15
    corr_df_sorted = corr_df.reindex(corr_df['r'].abs().sort_values(ascending=False).index)
    top_features = corr_df_sorted.head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top_features,
        y="feature",
        x="r",
        hue="feature",                     # temporarily use hue
        palette=["#E76F51" if r < 0 else "#2A9D8F" for r in top_features["r"]],
        legend=False                      # hide legend (since hue is not meaningful)
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Top {top_n} Features by Point-Biserial Correlation with Hostility")
    plt.xlabel("Point-Biserial Correlation (r)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


    return corr_df


import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np

def logistic_regression(df: pd.DataFrame, property_names):
    """
    Performs logistic regression on the selected features and the link type,
    and plots the most significant coefficients.
    """
    # Prepare target variable and binary feature
    df["label_bin"] = df["LINK_SENTIMENT"].replace({-1: 0, 1: 1})
    df["is_body"] = (df["link_source_type"] == "body").astype(int)

    X = df[property_names + ["is_body"]].copy()
    y = df["label_bin"]

    # Drop columns with NaN-only or constant values
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique() > 1]
    X = sm.add_constant(X)

    # Suppress convergence warnings temporarily
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            model = sm.Logit(y, X)
            result = model.fit(disp=False)
        except Exception as e:
            print("⚠️ Logistic regression failed to converge:", e)
            return None, None

    # Summary DataFrame
    summary_df = pd.DataFrame({
        "feature": result.params.index,
        "coef": result.params.values,
        "p_value": result.pvalues.values
    }).sort_values("p_value")

    print("\n=== Top 10 significant predictors ===")
    print(summary_df.head(10))

    # --- Plotting section ---
    top_n = 15
    summary_sorted = summary_df.reindex(summary_df["coef"].abs().sort_values(ascending=False).index)
    top_features = summary_sorted.head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        data=top_features,
        y="feature",
        x="coef",
        hue="feature",  # fix for Seaborn >= 0.13
        palette=["#E76F51" if c < 0 else "#2A9D8F" for c in top_features["coef"]],
        legend=False
    )
    plt.axvline(0, color="black", linewidth=1)
    plt.title(f"Top {top_n} Logistic Regression Coefficients")
    plt.xlabel("Coefficient (Effect on Probability of Non-Hostile Link)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return result, summary_df



def plot_feature_distributions(df: pd.DataFrame):
    """Displays basic exploratory feature distributions."""
    # --- Text length distribution ---
    plt.figure(figsize=(8,5))
    sns.kdeplot(
        data=df,
        x="n_words",
        hue="link_source_type",
        fill=True,
        common_norm=False,
        alpha=0.5
    )
    plt.xlim(0, 100)
    plt.title("Distribution of Text Length (in Words)")
    plt.xlabel("Number of Words")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.show()

    # --- Sentiment distribution by source type ---
    plt.figure(figsize=(6,5))
    sns.boxplot(
        data=df,
        x="link_source_type",
        y="VADER_compound",
        hue="link_source_type",   # fixes future warning
        palette="pastel",
        legend=False              # avoids duplicate legend
    )
    plt.title("Sentiment Distribution by Source Type")
    plt.ylabel("Compound Sentiment Score (VADER)")
    plt.tight_layout()
    plt.show()


def plot_top_subreddits(df: pd.DataFrame):
    """Plots the top 10 source and target subreddits by number of links."""
    
    # --- Top source subreddits ---
    top_sources = df["SOURCE_SUBREDDIT"].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=top_sources.values,
        y=top_sources.index,
        hue=top_sources.index,      # fixes future warning
        palette="crest",
        legend=False                # prevents redundant legend
    )
    plt.title("Top 10 Source Subreddits by Number of Links")
    plt.xlabel("Number of Links")
    plt.ylabel("Source Subreddit")
    plt.tight_layout()
    plt.show()

    # --- Top target subreddits ---
    top_targets = df["TARGET_SUBREDDIT"].value_counts().head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=top_targets.values,
        y=top_targets.index,
        hue=top_targets.index,      # fixes future warning
        palette="crest",
        legend=False
    )
    plt.title("Top 10 Target Subreddits by Number of Links")
    plt.xlabel("Number of Links")
    plt.ylabel("Target Subreddit")
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(df: pd.DataFrame, property_names):
    """Displays the correlation matrix between linguistic and sentiment features."""
    corr = df[property_names].corr()
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title("Correlation Matrix of Linguistic and Sentiment Properties")
    plt.show()
