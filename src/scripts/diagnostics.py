import pandas as pd

def check_missing_values(df: pd.DataFrame, name: str = "dataset"):
    """Displays the number of missing values per column and the total."""
    print(f"\n=== Missing values in {name} ===")
    print(df.isna().sum())
    print(f"Total NaNs: {df.isna().sum().sum()}")

def check_invalid_LINK_SENTIMENT(df: pd.DataFrame):
    """Checks that LINK_SENTIMENT contains only -1 and 1."""
    invalid_mask = ~df["LINK_SENTIMENT"].isin([-1, 1])
    invalid_rows = df[invalid_mask]
    print(f"Invalid LINK_SENTIMENT rows: {invalid_rows.shape[0]}")
    return invalid_rows

def check_empty_subreddits(df: pd.DataFrame, name: str):
    """Checks for empty subreddit fields (handles both hyperlink and embedding datasets)."""
    if {"SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"}.issubset(df.columns):
        bad_sources = df["SOURCE_SUBREDDIT"].astype(str).str.strip().eq("").sum()
        bad_targets = df["TARGET_SUBREDDIT"].astype(str).str.strip().eq("").sum()
        print(f"{name}: {bad_sources} empty SOURCE_SUBREDDIT, {bad_targets} empty TARGET_SUBREDDIT")
    elif "subreddit" in df.columns:
        bad_subs = df["subreddit"].astype(str).str.strip().eq("").sum()
        print(f"{name}: {bad_subs} empty subreddit")
    else:
        print(f"{name}: No subreddit-related columns found.")

