import numpy as np
import pandas as pd

# ============================
# Hostility augmentation
# ============================
def augment_hostility_df(df: pd.DataFrame, embeddings_subreddits: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df (named outside as df_hostility) with ONLY:
      - 'cosine_sim' column added (computed exactly as before)
      - 'is_neg' column added ((LINK_SENTIMENT < 0).astype(int))
    All intermediates (E, cos, link_embeddings_available) are kept inside.
    """
    out = df.copy()

    # Build normalized embedding matrix (same logic)
    E = embeddings_subreddits.set_index('subreddit').astype(float)
    E = E.div(np.linalg.norm(E.to_numpy(), axis=1, keepdims=True), axis=0)

    # Filter rows that have both embeddings (internal)
    link_embeddings_available = out[
        out['SOURCE_SUBREDDIT'].isin(E.index) &
        out['TARGET_SUBREDDIT'].isin(E.index)
    ].copy()

    # Cosine similarity for rows with both embeddings (internal)
    src = E.reindex(link_embeddings_available['SOURCE_SUBREDDIT']).to_numpy()
    tgt = E.reindex(link_embeddings_available['TARGET_SUBREDDIT']).to_numpy()
    cos = pd.Series(np.einsum('ij,ij->i', src, tgt), index=link_embeddings_available.index, name='cosine_sim')

    # Add back to the dataset (NaN where missing)
    out['cosine_sim'] = np.nan
    out.loc[cos.index, 'cosine_sim'] = cos

    # Add is_neg (exact logic)
    out['is_neg'] = (out['LINK_SENTIMENT'] < 0).astype(int)

    return out

# ============================
# Hidden Hostility / time-to-flip
# ============================
def build_followups(
    hyperlink_total: pd.DataFrame,
    *,
    window_time: str | pd.Timedelta,     # "24H" or pd.Timedelta(hours=24)
    label: str | None = None,
    drop_invalid_ts: bool = True
) -> pd.DataFrame:
    """
    For each origin link (LINK_SENTIMENT in {+1, -1}) at time ts1, look at the
    *next* link for the same unordered subreddit pair {A,B}.
    Keep the pair only if:
        (i) it's the immediate next event (forward in time),
        (ii) sentiment flips sign (s2 == -s1),
        (iii) 0 < (ts2 - ts1) <= window_time.

    Returns a tidy table with origin/match info and flip timing.
    """
    df = hyperlink_total.copy()

    # Ensure datetime; drop invalid if requested; sort stably by time
    if not np.issubdtype(df["TIMESTAMP"].dtype, np.datetime64):
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    if df["TIMESTAMP"].isna().any():
        if drop_invalid_ts:
            df = df.dropna(subset=["TIMESTAMP"])
        else:
            raise ValueError("Found invalid TIMESTAMP values (NaT).")

    df = df.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)

    # Unordered pair key so A→B and B→A are grouped together
    a = df["SOURCE_SUBREDDIT"].astype(str).to_numpy()
    b = df["TARGET_SUBREDDIT"].astype(str).to_numpy()
    df["pair_a"] = np.minimum(a, b)
    df["pair_b"] = np.maximum(a, b)

    # Within each unordered pair, grab the immediate NEXT event
    grp = df.sort_values("TIMESTAMP", kind="mergesort").groupby(["pair_a", "pair_b"], sort=False)

    df["next_src"] = grp["SOURCE_SUBREDDIT"].shift(-1)
    df["next_tgt"] = grp["TARGET_SUBREDDIT"].shift(-1)
    df["next_ts"]  = grp["TIMESTAMP"].shift(-1)
    df["next_s"]   = grp["LINK_SENTIMENT"].shift(-1)

    if isinstance(window_time, str):
        window_td = pd.Timedelta(window_time.lower())
    else:
        window_td = pd.Timedelta(window_time)
    dt = df["next_ts"] - df["TIMESTAMP"]

    mask = (
        df["next_ts"].notna() &
        (dt > pd.Timedelta(0)) &             # forward only
        (dt <= window_td) &                  # within window
        (df["next_s"] == -df["LINK_SENTIMENT"])  # sign flip (+1→-1 or -1→+1)
    )
    cand = df.loc[mask].copy()

    direction = np.where(
        (cand["next_src"].values == cand["SOURCE_SUBREDDIT"].values) &
        (cand["next_tgt"].values == cand["TARGET_SUBREDDIT"].values),
        "same (src→tgt)", "reverse (tgt→src)"
    )

    out = pd.DataFrame({
        "Origin_Source": cand["SOURCE_SUBREDDIT"].astype(str),
        "Origin_Target": cand["TARGET_SUBREDDIT"].astype(str),
        "Follow_Link_Direction": direction,
        "Timestamp1": cand["TIMESTAMP"],
        "Timestamp2": cand["next_ts"],
        "Time_Difference": dt.loc[cand.index].values,
        "Link_Sentiment1": cand["LINK_SENTIMENT"].astype(int),
        "Link_Sentiment2": cand["next_s"].astype(int),
        "Window_Label": label or f"time≤{window_td}",
    })

    return out

def windows_to_followups(
    hyperlink_total: pd.DataFrame,
    ) -> pd.DataFrame:
    windows = [
        ("10H", "10H", 10.0),
        ("24H", "24H", 24.0),
        ("2D",  "2D",  48.0),
        ("7D",  "7D",  168.0),
    ]

    followups_dict = {}
    for label, w, hours in windows:
        f = build_followups(hyperlink_total, window_time=w, label=f"time={w}")
        followups_dict[label] = f
    return followups_dict
