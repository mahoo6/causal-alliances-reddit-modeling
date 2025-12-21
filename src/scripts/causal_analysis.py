import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from IPython.display import display
from scipy.stats import binomtest
from scipy.stats import binom


import pandas as pd

def add_month_index(df_monthly, month_col="month"):
    """
    Standardize month as 'YYYY-MM' and add month_idx = 1..T.
    Also drops POST_ID and TIMESTAMP if present.
    
    Returns:
      df_monthly_out (with 'month' as str and 'month_idx' as int),
      months_sorted, month_to_idx, idx_to_month
    """
    df = df_monthly.copy()

    # Drop unnecessary post-level identifiers if present
    df = df.drop(columns=["POST_ID", "TIMESTAMP"], errors="ignore")

    # Standardize month format
    df[month_col] = (
        pd.to_datetime(df[month_col], errors="coerce")
          .dt.to_period("M")
          .astype(str)
    )

    # Build mapping
    months_sorted = sorted(df[month_col].dropna().unique())
    month_to_idx = {m: i + 1 for i, m in enumerate(months_sorted)}
    idx_to_month = {i + 1: m for i, m in enumerate(months_sorted)}

    # Attach month_idx
    df["month_idx"] = df[month_col].map(month_to_idx).astype(int)

    print("Total months:", len(months_sorted))
    print("First months:", months_sorted[:5])

    return df, months_sorted, month_to_idx, idx_to_month

import numpy as np
import pandas as pd

def build_monthly_counts(edges: pd.DataFrame):
    pm = (
        edges
        .groupby(["month_idx", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"])
        .size()
        .rename("n")
        .reset_index()
        .pivot(
            index=["month_idx", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"],
            columns="LINK_SENTIMENT",
            values="n"
        )
        .fillna(0)
        .rename(columns={-1: "neg", +1: "pos"})
        .reset_index()
    )

    # remove unwanted column index name
    pm.columns.name = None

    if "pos" not in pm:
        pm["pos"] = 0
    if "neg" not in pm:
        pm["neg"] = 0

    # rename month_idx → month
    pm = pm.rename(columns={"month_idx": "month"})

    sm = (
        pm.groupby(["month", "SOURCE_SUBREDDIT"])[["pos", "neg"]]
        .sum()
        .reset_index()
        .rename(columns={
            "SOURCE_SUBREDDIT": "subreddit",
            "pos": "out_pos",
            "neg": "out_neg"
        })
    )

    sm["out_total"] = sm["out_pos"] + sm["out_neg"]

    return pm, sm



def build_monthly_unordered_pair_scores(df_monthly: pd.DataFrame):
    """
    Build a table with one row per (month, A, B) unordered pair (A < B),
    restricted to pairs that interacted at least once in that month.
    
    Returns columns:
      month, A, B, count, sum, Pos_Links, Neg_Links, Ratio, Friendship_Score
    """
    df = df_monthly.copy()

    # remove self-links
    df = df[df["SOURCE_SUBREDDIT"] != df["TARGET_SUBREDDIT"]].copy()

    # build unordered pair keys
    df["A"] = df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].min(axis=1)
    df["B"] = df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].max(axis=1)

    # aggregate per (month_idx, A, B)
    g = (
        df.groupby(["month_idx", "A", "B"])["LINK_SENTIMENT"]
          .agg(count="size", sum="sum")
          .reset_index()
    )

    # rename month_idx -> month
    g = g.rename(columns={"month_idx": "month"})

    # derive pos/neg counts and score
    g["Pos_Links"] = (g["count"] + g["sum"]) / 2
    g["Neg_Links"] = (g["count"] - g["sum"]) / 2
    g["Ratio"] = g["sum"] / g["count"]
    g["Friendship_Score"] = g["Ratio"] * np.log1p(g["count"])

    return g.sort_values(["month", "A", "B"]).reset_index(drop=True)


from sklearn.cluster import KMeans
import numpy as np

def learn_friend_enemy_thresholds(pair_monthly_scores: pd.DataFrame,
                                  score_col: str = "Friendship_Score",
                                  random_state: int = 42):
    """
    Learn enemy and friendship thresholds from pooled monthly Friendship_Score
    using K-Means (k=3).
    
    Returns:
      ENEMY_THRESHOLD, FRIEND_THRESHOLD, centers
    """
    scores = (
        pair_monthly_scores[score_col]
        .dropna()
        .values
        .reshape(-1, 1)
    )

    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    kmeans.fit(scores)

    centers = np.sort(kmeans.cluster_centers_.flatten())

    ENEMY_THRESHOLD = (centers[0] + centers[1]) / 2
    FRIEND_THRESHOLD = (centers[1] + centers[2]) / 2

    return ENEMY_THRESHOLD, FRIEND_THRESHOLD, centers


def classify_monthly_relationship(score,
                                  enemy_threshold,
                                  friend_threshold):
    if score < enemy_threshold:
        return "Enemy"
    elif score > friend_threshold:
        return "Friend"
    else:
        return "Neutral"


import numpy as np
import pandas as pd
from itertools import combinations

def build_pair_event_panel_from_df(df_monthly: pd.DataFrame, T: int):
    """
    Build pair_event_panel_all directly from df_monthly using month_idx.

    Returns a wide DataFrame with:
      columns: ['C','A','B','month_1',...,'month_T']
      where each month cell is (attacks_A, attacks_B) if both attacked C that month, else NaN.
    """
    df = df_monthly.copy()

    # keep only negative links (attacks)
    df = df[df["LINK_SENTIMENT"] == -1].copy()

    # count attacks per (month_idx, C, attacker)
    ac = (
        df.groupby(["month_idx", "TARGET_SUBREDDIT", "SOURCE_SUBREDDIT"])
          .size()
          .rename("attacks")
          .reset_index()
          .rename(columns={"TARGET_SUBREDDIT": "C", "SOURCE_SUBREDDIT": "attacker"})
    )

    records = []

    # iterate per (month, C) to generate co-attacker pairs
    for (m, C), g in ac.groupby(["month_idx", "C"]):
        attackers = g["attacker"].to_numpy()
        if len(attackers) < 2:
            continue

        atk_counts = dict(zip(g["attacker"], g["attacks"]))

        for A, B in combinations(sorted(attackers), 2):
            records.append({
                "C": C,
                "A": A,
                "B": B,
                "month_idx": int(m),
                "pair_val": (int(atk_counts[A]), int(atk_counts[B]))
            })

    if not records:
        # empty structure but correct columns
        cols = ["C", "A", "B"] + [f"month_{i}" for i in range(1, T + 1)]
        return pd.DataFrame(columns=cols)

    df_rec = pd.DataFrame(records)

    panel = df_rec.pivot_table(
        index=["C", "A", "B"],
        columns="month_idx",
        values="pair_val",
        aggfunc="first"
    )

    panel.columns = [f"month_{int(c)}" for c in panel.columns]
    panel = panel.reset_index()

    # ensure all months exist
    for i in range(1, T + 1):
        col = f"month_{i}"
        if col not in panel.columns:
            panel[col] = np.nan

    ordered_cols = ["C", "A", "B"] + [f"month_{i}" for i in range(1, T + 1)]
    panel = panel[ordered_cols].sort_values(["C", "A", "B"]).reset_index(drop=True)

    return panel



def build_enemy_status_lookup(pair_monthly_scores: pd.DataFrame):
    """
    Fast lookup for monthly relationship status between unordered pairs.

    Expects columns: month, A, B, status
    Returns: dict keyed by (month, A, B) -> status
    """
    needed = pair_monthly_scores[["month", "A", "B", "status"]].copy()
    return {
        (int(r.month), r.A, r.B): r.status
        for r in needed.itertuples(index=False)
    }


def filter_pair_event_panel_by_enemy_status(pair_event_panel_all: pd.DataFrame,
                                           enemy_lookup: dict):
    """
    Keep only months where:
      (i)  A and B both attacked C that month  -> tuple exists AND both counts > 0
      (ii) A is Enemy of C AND B is Enemy of C that month (monthly status)

    All other month cells are set to NaN.
    Rows with no remaining months are dropped.
    """
    panel = pair_event_panel_all.copy()
    month_cols = [c for c in panel.columns if c.startswith("month_")]

    def get_status(month_idx: int, X: str, Y: str):
        A_, B_ = sorted([X, Y])
        return enemy_lookup.get((month_idx, A_, B_), None)

    def keep_month_val(val, month_idx: int, A: str, B: str, C: str):
        # must be a tuple with BOTH attack counts > 0
        if not isinstance(val, tuple):
            return np.nan
        a_cnt, b_cnt = val
        if not (a_cnt > 0 and b_cnt > 0):
            return np.nan

        # must be enemies with C for this month
        s_AC = get_status(month_idx, A, C)
        s_BC = get_status(month_idx, B, C)
        if s_AC == "Enemy" and s_BC == "Enemy":
            return (int(a_cnt), int(b_cnt))
        return np.nan

    # apply per row
    def filter_row(row):
        C, A, B = row["C"], row["A"], row["B"]
        for mcol in month_cols:
            month_idx = int(mcol.split("_")[1])
            row[mcol] = keep_month_val(row[mcol], month_idx, A, B, C)
        return row

    panel = panel.apply(filter_row, axis=1)

    # drop rows with no remaining months (no true-conflict months)
    def has_any_tuple(r):
        return any(isinstance(r[c], tuple) for c in month_cols)

    panel = panel.loc[panel.apply(has_any_tuple, axis=1)].reset_index(drop=True)

    return panel


def build_pair_summary(pair_event_panel: pd.DataFrame):
    month_cols = [c for c in pair_event_panel.columns if c.startswith("month_")]

    def coattack_stats(row):
        start = np.nan
        end = np.nan
        active = 0
        for i, col in enumerate(month_cols, start=1):
            val = row[col]
            if isinstance(val, tuple):
                if np.isnan(start):
                    start = i
                end = i
                active += 1
        return pd.Series({
            "start_month": start,
            "end_month": end,
            "active_months": active
        })

    pair_summary = pair_event_panel[["C", "A", "B"]].copy()
    pair_summary[["start_month", "end_month", "active_months"]] = pair_event_panel.apply(coattack_stats, axis=1)
    return pair_summary


import numpy as np
import pandas as pd

def build_monthly_score_lookup(pair_monthly_scores: pd.DataFrame, score_col: str = "Friendship_Score"):
    """
    Build fast lookup: (month, A, B) -> Friendship_Score
    Assumes (A,B) are unordered keys and month is the monthly index.
    """
    needed = pair_monthly_scores[["month", "A", "B", score_col]].copy()
    return {
        (int(r.month), r.A, r.B): float(getattr(r, score_col))
        for r in needed.itertuples(index=False)
    }


def build_pair_friendship_score_panel_all_months(pair_event_panel: pd.DataFrame,
                                                 months_sorted: list,
                                                 score_lookup: dict):
    """
    For each (C,A,B) row in pair_event_panel, create month_1..month_T columns
    containing FriendshipScore(A,B,t) for EVERY month t=1..T.

    If a pair (A,B) has no score recorded in a month (no interactions),
    the value is NaN (since the monthly score table only includes interacted pairs).
    """
    base = pair_event_panel[["C", "A", "B"]].copy()
    T = len(months_sorted)

    def get_score(month_idx: int, A: str, B: str):
        A_, B_ = sorted([A, B])
        return score_lookup.get((month_idx, A_, B_), np.nan)

    for t in range(1, T + 1):
        base[f"month_{t}"] = base.apply(lambda r: get_score(t, r["A"], r["B"]), axis=1)

    return base


def build_friendship_stat_from_score_panel(friendship_score_panel: pd.DataFrame,
                                          FRIEND_THRESHOLD: float):
    """
    Build friendship timeline stats per (A,B) based on score > FRIEND_THRESHOLD.

    Returns columns:
      A, B, start_month, end_month, active_months
    """
    month_cols = [c for c in friendship_score_panel.columns if c.startswith("month_")]

    # If the same (A,B) appears for multiple C, collapse by taking max score per month
    collapsed_rows = []
    for (A, B), g in friendship_score_panel.groupby(["A", "B"]):
        rec = {"A": A, "B": B}
        for mcol in month_cols:
            vals = g[mcol].to_numpy(dtype=float)
            rec[mcol] = np.nanmax(vals) if np.any(~np.isnan(vals)) else np.nan
        collapsed_rows.append(rec)

    collapsed = pd.DataFrame(collapsed_rows)

    def stat_row(row):
        friend_months = []
        for mcol in month_cols:
            month_idx = int(mcol.split("_")[1])
            val = row[mcol]
            if isinstance(val, (int, float, np.floating)) and not np.isnan(val):
                if val > FRIEND_THRESHOLD:
                    friend_months.append(month_idx)

        if len(friend_months) == 0:
            return pd.Series({"start_month": 0, "end_month": 0, "active_months": 0})

        return pd.Series({
            "start_month": int(min(friend_months)),
            "end_month": int(max(friend_months)),
            "active_months": int(len(friend_months))
        })

    friendship_stat = collapsed[["A", "B"]].copy()
    friendship_stat[["start_month", "end_month", "active_months"]] = collapsed.apply(stat_row, axis=1)

    return friendship_stat.sort_values(["A", "B"]).reset_index(drop=True)



def build_conflict_friendship_comparison_score_based(
    pair_summary: pd.DataFrame,
    friendship_stat: pd.DataFrame
) -> pd.DataFrame:
    """
    Minimal-change version of the old comparison builder, but using:
      - pair_summary: (C,A,B) with start_month/end_month from enemy-based co-attacks
      - friendship_stat: (A,B) with start_month (friendship) from score thresholding

    Returns columns:
      C, A, B, conflict_start, conflict_end, friendship_start,
      friendship_observed, friendship_after, new_friendship, far_friendship
    """

    # conflict timing maps (per C,A,B)
    conflict_start_map = pair_summary.set_index(["C", "A", "B"])["start_month"].to_dict()
    conflict_end_map   = pair_summary.set_index(["C", "A", "B"])["end_month"].to_dict()

    # friendship start map (per A,B)
    friendship_start_map = friendship_stat.set_index(["A", "B"])["start_month"].to_dict()

    out_rows = []
    for r in pair_summary[["C", "A", "B"]].itertuples(index=False):
        C, A, B = r.C, r.A, r.B

        conflict_start = conflict_start_map.get((C, A, B), np.nan)
        conflict_end   = conflict_end_map.get((C, A, B), np.nan)

        friendship_start = int(friendship_start_map.get((A, B), 0))  # 0 if never friends
        friendship_observed = (friendship_start > 0)

        friendship_after = (
            friendship_observed
            and (not pd.isna(conflict_start))
            and (friendship_start >= int(conflict_start))
        )

        if friendship_observed and (not pd.isna(conflict_start)) and (not pd.isna(conflict_end)):
            new_friendship = (int(conflict_start) <= friendship_start <= int(conflict_end) + 1)
        else:
            new_friendship = False

        if friendship_observed and (not pd.isna(conflict_end)):
            far_friendship = (friendship_start > int(conflict_end) + 1)
        else:
            far_friendship = False

        out_rows.append({
            "C": C,
            "A": A,
            "B": B,
            "conflict_start": conflict_start,
            "conflict_end": conflict_end,
            "friendship_start": friendship_start,
            "friendship_observed": friendship_observed,
            "friendship_after": friendship_after,
            "new_friendship": new_friendship,
            "far_friendship": far_friendship
        })

    return pd.DataFrame(out_rows).sort_values(["C", "A", "B"]).reset_index(drop=True)




def build_treated_pairs_from_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Build treated_pairs from the comparison table (one row per (C,A,B)).

    Steps (same logic as your code):
      1) Drop "old friends": friendship_start > 0 and friendship_start < conflict_start
      2) Drop rows with missing conflict_start/conflict_end
      3) Collapse to one row per (A,B):
           conflict_start = min(conflict_start)
           conflict_end   = max(conflict_end)
      4) Add duration and treated=1
    """
    cs = comparison.copy()

    # 1) Remove "old friends"
    old_friends_mask = (cs["friendship_start"] > 0) & (cs["friendship_start"] < cs["conflict_start"])
    print("Rows flagged as old friends (to drop):", old_friends_mask.sum())
    cs_clean = cs.loc[~old_friends_mask].copy()

    # Precaution: remove ill-defined conflict windows
    cs_clean = cs_clean.dropna(subset=["conflict_start", "conflict_end"])

    # 2) Collapse to one row per pair (A,B)
    def summarize_pair(group: pd.DataFrame) -> pd.Series:
        conflict_start = int(group["conflict_start"].min())
        conflict_end   = int(group["conflict_end"].max())
        return pd.Series({"conflict_start": conflict_start, "conflict_end": conflict_end})

    treated_pairs = (
        cs_clean
        .groupby(["A", "B"], as_index=False)
        .apply(summarize_pair)
        .reset_index(drop=True)
    )

    # 3) Add duration and treated flag
    treated_pairs["duration"] = treated_pairs["conflict_end"] - treated_pairs["conflict_start"] + 1
    treated_pairs["treated"] = 1

    print("Final treated_pairs shape:", treated_pairs.shape)
    return treated_pairs


import numpy as np
import pandas as pd

def build_global_friendship_stat_score_based(pair_monthly_scores: pd.DataFrame):
    """
    Score-based global friendship stats for ALL unordered pairs (A,B) that ever interacted.

    Expects pair_monthly_scores columns: month, A, B, status
      - month is an integer month index (1..T)
      - status in {"Friend","Neutral","Enemy"} (or similar)

    Returns:
      friendship_stat_all with columns:
        A, B, start_month, end_month, active_months
      where:
        start_month = first month with status == "Friend" (else 0)
        end_month   = last  month with status == "Friend" (else 0)
        active_months = number of months with status == "Friend"
    """
    df = pair_monthly_scores[["month", "A", "B", "status"]].copy()

    # keep only friend months
    f = df[df["status"] == "Friend"].copy()

    if f.empty:
        # no friends at all
        friendship_stat_all = (
            df[["A", "B"]].drop_duplicates()
            .assign(start_month=0, end_month=0, active_months=0)
            .reset_index(drop=True)
        )
        return friendship_stat_all

    # stats from friend months
    stats = (
        f.groupby(["A", "B"])["month"]
         .agg(start_month="min", end_month="max", active_months="count")
         .reset_index()
    )

    # include pairs that never become friends but do exist in the universe
    universe = df[["A", "B"]].drop_duplicates()
    friendship_stat_all = universe.merge(stats, on=["A", "B"], how="left").fillna(0)

    # ensure int types for months
    friendship_stat_all["start_month"] = friendship_stat_all["start_month"].astype(int)
    friendship_stat_all["end_month"] = friendship_stat_all["end_month"].astype(int)
    friendship_stat_all["active_months"] = friendship_stat_all["active_months"].astype(int)

    return friendship_stat_all







import numpy as np
import pandas as pd
from numpy.linalg import norm



# =====================================================
# 3.1 Pair-level activity
# =====================================================
def add_pair_activity_pre(df_pairs, sub_month):
    """
    Adds 'activity' = log(1 + total outgoing links of A and B 
    BEFORE the conflict_start for that pair).
    
    """

    # Build panel: (subreddit, month) → total outgoing links that month
    panel = (
        sub_month
        .set_index(["subreddit", "month"])["out_total"]
        .unstack(fill_value=0)
    )
    
    # Ensure columns are sorted month_1...month_T
    panel = panel.reindex(sorted(panel.columns), axis=1).fillna(0)
    
    # Now compute activity for each pair row
    def compute_activity(row):
        A, B = row["A"], row["B"]
        cs = int(row["conflict_start"])
        
        # months strictly before conflict_start
        valid_months = [m for m in panel.columns if m < cs]
        
        # sum outgoing of A and B before conflict-start
        outA = panel.loc[A, valid_months].sum() if A in panel.index else 0
        outB = panel.loc[B, valid_months].sum() if B in panel.index else 0
        
        return np.log1p(outA + outB)
    
    df_pairs["activity"] = df_pairs.apply(compute_activity, axis=1)
    return df_pairs



# =====================================================
# 3.2 Pair-level aggressiveness
# =====================================================
def add_pair_aggressiveness_pre(df_pairs, sub_month):
    """
    Adds 'aggressiveness' = mean pre-conflict aggressiveness of A and B.
    
    For each subreddit X:
        aggr_pre(X) = sum(out_neg before cs) / sum(out_total before cs)
    where cs = conflict_start for that pair.
    """
    

    sm = sub_month.copy()
    
    # Build monthly OUT panel
    out_panel = (
        sm.set_index(["subreddit", "month"])["out_total"]
          .unstack(fill_value=0)
    )
    out_panel = out_panel.reindex(sorted(out_panel.columns), axis=1).fillna(0)
    
    # Build monthly NEG panel
    neg_panel = (
        sm.set_index(["subreddit", "month"])["out_neg"]
          .unstack(fill_value=0)
    )
    neg_panel = neg_panel.reindex(sorted(neg_panel.columns), axis=1).fillna(0)
    
    def compute_aggr(row):
        A, B = row["A"], row["B"]
        cs = int(row["conflict_start"])
        
        # months strictly < conflict start
        valid_months = [m for m in out_panel.columns if m < cs]
        
        # For subreddit A
        if A in out_panel.index:
            totalA = out_panel.loc[A, valid_months].sum()
            negA   = neg_panel.loc[A, valid_months].sum()
            agA    = (negA / totalA) if totalA > 0 else 0
        else:
            agA = 0
        
        # For subreddit B
        if B in out_panel.index:
            totalB = out_panel.loc[B, valid_months].sum()
            negB   = neg_panel.loc[B, valid_months].sum()
            agB    = (negB / totalB) if totalB > 0 else 0
        else:
            agB = 0
        
        # Pair-level aggressiveness
        return (agA + agB) / 2
    
    df_pairs["aggressiveness"] = df_pairs.apply(compute_aggr, axis=1)
    return df_pairs


# =====================================================
# 3.3 Pair-level cosine similarity
# =====================================================
def add_pair_similarity(df_pairs, emb_df):
    emb_lookup = {
        row.subreddit: row.iloc[1:].to_numpy(dtype=float)
        for _, row in emb_df.iterrows()
    }

    def cosine(a, b):
        if a is None or b is None:
            return 0.0
        na, nb = norm(a), norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def compute_sim(row):
        eA = emb_lookup.get(row["A"], None)
        eB = emb_lookup.get(row["B"], None)
        return cosine(eA, eB)

    df_pairs["similarity"] = df_pairs.apply(compute_sim, axis=1)
    return df_pairs


# =====================================================
# 3.4 Pre-conflict hostility
# =====================================================
def add_preconflict_hostility(df_pairs, pair_month):
    """
    Adds hostility_pre:
      = total negative links exchanged between A and B before conflict_start.
    
    """

    # ---------------------------------------------------
    # Build fast lookup: neg_lookup[(src, tgt)][month] = neg count
    # ---------------------------------------------------
    neg_lookup = {}

    for _, r in pair_month.iterrows():
        src = r["SOURCE_SUBREDDIT"]
        tgt = r["TARGET_SUBREDDIT"]
        m_idx = int(r["month"])
        neg = r["neg"]

        neg_lookup.setdefault((src, tgt), {})[m_idx] = neg


    # ---------------------------------------------------
    # Compute hostility_pre for each (A,B)
    # ---------------------------------------------------
    def compute_hostility(row):
        A, B = row["A"], row["B"]
        t0 = int(row["conflict_start"])

        total_h = 0

        # Sum over all months strictly before t0
        for m_idx in range(1, t0):
            # A → B direction
            if (A, B) in neg_lookup:
                total_h += neg_lookup[(A, B)].get(m_idx, 0)

            # B → A direction
            if (B, A) in neg_lookup:
                total_h += neg_lookup[(B, A)].get(m_idx, 0)

        return total_h

    df_pairs["hostility_pre"] = df_pairs.apply(compute_hostility, axis=1)
    return df_pairs


# =====================================================
# 3.5 MASTER FUNCTION — Apply all confounders
# =====================================================
def add_all_confounders(treated_pairs, control_pairs, sub_month, emb_df, pair_month):

    # ---- treated ----
    treated_pairs = add_pair_activity_pre(treated_pairs, sub_month)
    treated_pairs = add_pair_aggressiveness_pre(treated_pairs, sub_month)
    treated_pairs = add_pair_similarity(treated_pairs, emb_df)
    treated_pairs = add_preconflict_hostility(treated_pairs, pair_month)


    # ---- control ----
    control_pairs = add_pair_activity_pre(control_pairs, sub_month)
    control_pairs = add_pair_aggressiveness_pre(control_pairs, sub_month)
    control_pairs = add_pair_similarity(control_pairs, emb_df)
    control_pairs = add_preconflict_hostility(control_pairs, pair_month)

    return treated_pairs, control_pairs



# ============================================================
# STEP 4 — PROPENSITY SCORE ESTIMATION
# ============================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# ------------------------------------------------------------
# 4.1 Combine treated and control into a single modeling frame
# ------------------------------------------------------------
def build_ps_dataset(treated_pairs_conf, control_pairs_conf):
    """
    Concatenate treated and control tables; keep only
    necessary columns for propensity-score modeling.
    """
    treated = treated_pairs_conf.copy()
    control = control_pairs_conf.copy()

    df_mix = pd.concat([treated, control], ignore_index=True)

    # Columns used as features for propensity score estimation
    confounders = ["activity", "aggressiveness", "similarity", "hostility_pre"]
    
    X = df_mix[confounders].copy()
    y = df_mix["treated"].astype(int)

    return df_mix, X, y, confounders


# ------------------------------------------------------------
# 4.2 Fit logistic regression with scaling
# ------------------------------------------------------------
def fit_propensity_score_model(X, y):
    """
    Scales inputs and fits logistic regression on top.
    Returns fitted scaler, model, and scaled X.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logit = LogisticRegression(
        penalty=None, 
        solver="lbfgs",
        max_iter=2000
    )

    logit.fit(X_scaled, y)

    return scaler, logit, X_scaled


# ------------------------------------------------------------
# 4.3 Compute propensity scores and attach to df
# ------------------------------------------------------------
def compute_propensity_scores(df, scaler, logit, confounders):
    X = df[confounders].copy()
    X_scaled = scaler.transform(X)
    df["pscore"] = logit.predict_proba(X_scaled)[:, 1]
    return df






# STEP 5 — PROPENSITY SCORE MATCHING

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------
# 5.1 Identify common support region
# ------------------------------------------------------------
def apply_common_support(df_ps):
    """
    Remove treated and control units outside the overlapping
    propensity score region.
    """
    treated_scores  = df_ps.loc[df_ps.treated == 1, "pscore"]
    control_scores  = df_ps.loc[df_ps.treated == 0, "pscore"]

    min_T, max_T = treated_scores.min(), treated_scores.max()
    min_C, max_C = control_scores.min(), control_scores.max()

    # boundaries of the overlap
    lower = max(min_T, min_C)
    upper = min(max_T, max_C)

    print(f"Common support interval: [{lower:.4f}, {upper:.4f}]")

    # keep only within this region
    mask = (df_ps["pscore"] >= lower) & (df_ps["pscore"] <= upper)
    df_filtered = df_ps.loc[mask].copy()

    # report counts
    print("Treated before common support:", (df_ps.treated == 1).sum())
    print("Control before common support:", (df_ps.treated == 0).sum())
    print("Treated after common support:",  (df_filtered.treated == 1).sum())
    print("Control after common support:",  (df_filtered.treated == 0).sum())

    return df_filtered, lower, upper


# ------------------------------------------------------------
# 5.2 Nearest-neighbor matching (1:1, with replacement)
# ------------------------------------------------------------
def nearest_neighbor_match(df_filtered):
    """
    Performs nearest-neighbor matching using propensity scores.
    Returns a DataFrame of matched treated–control rows.
    """
    treated  = df_filtered[df_filtered.treated == 1].copy().reset_index(drop=True)
    control  = df_filtered[df_filtered.treated == 0].copy().reset_index(drop=True)

    # Fit NN model on control group propensity scores
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(control[["pscore"]])

    # Find nearest control for each treated
    distances, indices = neigh.kneighbors(treated[["pscore"]])

    treated["match_index"]  = indices.flatten()
    treated["match_dist"]   = distances.flatten()

    # Retrieve matched control rows
    matched_controls = control.iloc[treated["match_index"]].copy()
    matched_controls = matched_controls.reset_index(drop=True)
    
    # Add match IDs
    treated["matched_pair_id"]  = np.arange(len(treated))
    matched_controls["matched_pair_id"] = treated["matched_pair_id"]

    # Combine into final matched set
    matched_df = pd.concat([treated, matched_controls], ignore_index=True)
    matched_df = matched_df.sort_values("matched_pair_id").reset_index(drop=True)

    print("Number of matched pairs:", len(treated))

    return matched_df


# ------------------------------------------------------------
# 5.3 Full wrapper function
# ------------------------------------------------------------
def run_matching(df_ps):
    df_filtered, lower, upper = apply_common_support(df_ps)
    matched_df = nearest_neighbor_match(df_filtered)
    return matched_df




# ============================================================
# STEP 6 — PART 1: Add Y directly to matched_df
# ============================================================

import numpy as np
import pandas as pd



# ------------------------------------------------------------
# 6.1 Add Y to matched_df
# ------------------------------------------------------------
def add_outcome_to_matched(matched_df, friendship_lookup):
    
    df = matched_df.copy()

    def lookup_start(r):
        A_, B_ = sorted([r.A, r.B])
        return int(friendship_lookup.get((A_, B_), 0))

    df["friendship_start"] = df.apply(lookup_start, axis=1)

    # For treated: conflict_start / conflict_end come from real co-attacks
    # For control: conflict_start / conflict_end are pseudo windows
    # In both cases, the rule is identical.
    df["Y"] = (
        (df["friendship_start"] > 0) &
        (df["friendship_start"] >= df["conflict_start"]) &
        (df["friendship_start"] <= df["conflict_end"] + 1)
    ).astype(int)

    df = df.drop(columns=["friendship_start"])
    return df





# ============================================================
#  ATT FOR 1:1 MATCHING — PAIRWISE DIFFERENCES
# ============================================================

def att_pairwise(matched_with_Y):
    """
    Computes ATT by averaging (Y_treated - Y_control)
    within each matched pair.
    """
    diffs = []

    for pid, group in matched_with_Y.groupby("matched_pair_id"):
        # Expect exactly 1 treated and 1 control
        yt = group[group.treated == 1]["Y"]
        yc = group[group.treated == 0]["Y"]

        if len(yt) == 1 and len(yc) == 1:
            diffs.append(float(yt.values[0] - yc.values[0]))
        else:
            # In rare mismatches, skip the pair
            continue

    att = np.mean(diffs)

    print("=== PAIRWISE ATT ESTIMATE ===")
    print(f"Number of matched pairs: {len(diffs)}")
    print(f"ATT = {att*100:.2f} percentage points")

    return att, diffs



import numpy as np

def bootstrap_att(diffs, n_boot=5000, ci=95, seed=42):
    """
    Bootstrap CI for ATT from matched-pair differences.
    
    diffs: list/array of Δ_i = Y_treated - Y_control for each matched pair
    n_boot: number of bootstrap resamples
    ci: confidence level (e.g., 95)
    """
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    rng = np.random.default_rng(seed)

    # Bootstrap distribution of the mean
    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)  # resample pairs
        boot_means[b] = sample.mean()

    att_hat = diffs.mean()
    alpha = (100 - ci) / 2
    lo = np.percentile(boot_means, alpha)
    hi = np.percentile(boot_means, 100 - alpha)

    # Bootstrap SE
    se = boot_means.std(ddof=1)

    # p-value for H0: ATT <= 0
    # (approximate, using bootstrap distribution centered at 0)
    p_value = (boot_means <= 0).mean()

    print("=== BOOTSTRAP ATT INFERENCE ===")
    print(f"ATT (point estimate): {att_hat*100:.2f} percentage points")
    print(f"Bootstrap standard deviation: {se*100:.2f} pp")
    print(f"{ci}% CI (percentile): [{lo*100:.2f}, {hi*100:.2f}] pp")
    print(f"Approx. p-value for H0: ATT<=0: {p_value:.4f}")

    return {
        "att": att_hat,
        "boot_means": boot_means,
        "se": se,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": p_value
    }





def build_friend_lookup_all(friendship_stat_all: pd.DataFrame):
    """
    Build lookup dictionary mapping unordered pairs (A,B)
    to their friendship start month.

    Returns:
      dict: (A, B) -> start_month
    """
    return {
        (row.A, row.B): int(row.start_month)
        for row in friendship_stat_all.itertuples(index=False)
    }



def build_control_pairs(
    df_monthly: pd.DataFrame,
    treated_pairs: pd.DataFrame,
    pair_event_panel_all: pd.DataFrame,
    friendship_stat_all: pd.DataFrame,
    seed: int = 42
):
    """
    Build control pairs (never-treated) and assign pseudo conflict windows.

    Steps (exactly as in the notebook):
      1) Build universe of unordered pairs (A,B) with at least one link
      2) Remove treated pairs
      3) Remove any pair that ever co-attacked (weak or strong)
      4) Sample pseudo conflict windows from treated conflict_start and duration
      5) Remove control pairs already friends before pseudo_start

    Returns:
      control_pairs (DataFrame)
    """

    # -----------------------------
    # convenience: total number of months (needed for clipping instead the generated conflict end exceeds month 41)
    # -----------------------------
    T = int(df_monthly["month_idx"].max())

    # -----------------------------
    # 2.1 Build universe of unordered pairs with at least one link
    # -----------------------------
    links_df = df_monthly[df_monthly["SOURCE_SUBREDDIT"] != df_monthly["TARGET_SUBREDDIT"]].copy()
    links_df["A"] = links_df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].min(axis=1)
    links_df["B"] = links_df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT"]].max(axis=1)

    pair_universe = links_df[["A", "B"]].drop_duplicates().reset_index(drop=True)
    print("Total (A,B) pairs with at least one link:", pair_universe.shape[0])

    # -----------------------------
    # 2.2 Remove treated pairs
    # -----------------------------
    treated_index = set(map(tuple, treated_pairs[["A", "B"]].to_numpy()))
    pair_universe = pair_universe[~pair_universe.set_index(["A", "B"]).index.isin(treated_index)].reset_index(drop=True)
    print("After removing treated pairs:", pair_universe.shape[0])

    # -----------------------------
    # 2.3 Remove any pair that ever co-attacked (weak or strong)
    # -----------------------------
    coattack_pairs = pair_event_panel_all[["A", "B"]].drop_duplicates()
    coattack_index = set(map(tuple, coattack_pairs.to_numpy()))

    pair_universe = pair_universe[~pair_universe.set_index(["A", "B"]).index.isin(coattack_index)].reset_index(drop=True)
    print("After removing any co-attacking pairs:", pair_universe.shape[0])

    # -----------------------------
    # 2.4 Assign pseudo conflict windows from treated distributions
    # -----------------------------
    rng = np.random.default_rng(seed)

    N_controls = pair_universe.shape[0]
    treated_starts = treated_pairs["conflict_start"].to_numpy()
    treated_durs   = treated_pairs["duration"].to_numpy()

    pseudo_starts = rng.choice(treated_starts, size=N_controls, replace=True)
    pseudo_durs   = rng.choice(treated_durs,   size=N_controls, replace=True)

    pseudo_ends = pseudo_starts + pseudo_durs - 1
    pseudo_ends = np.minimum(pseudo_ends, T)           # clip
    pseudo_durs_eff = pseudo_ends - pseudo_starts + 1  # recompute

    control_pairs = pair_universe.copy()
    control_pairs["conflict_start"] = pseudo_starts
    control_pairs["conflict_end"]   = pseudo_ends
    control_pairs["duration"]       = pseudo_durs_eff

    print("Control pairs before filtering already-friends:", control_pairs.shape[0])

    # -----------------------------
    # 2.5 Remove control pairs that are already Friends before pseudo_start
    #     (score-based start month lookup)
    # -----------------------------
    fs = friendship_stat_all.rename(columns={"start_month": "friend_start"})[["A", "B", "friend_start"]]
    control_pairs = control_pairs.merge(fs, on=["A", "B"], how="left")

    old_friends_mask_ctrl = (
        control_pairs["friend_start"].notna()
        & (control_pairs["friend_start"] > 0)
        & (control_pairs["friend_start"] < control_pairs["conflict_start"])
    )

    print("Control pairs flagged as already friends (to drop):", old_friends_mask_ctrl.sum())

    control_pairs = control_pairs[~old_friends_mask_ctrl].copy()
    control_pairs["treated"] = 0

    control_pairs = control_pairs.drop(columns=["friend_start"])

    print("Final control_pairs shape:", control_pairs.shape)

    return control_pairs




def basic_sign_test_stats(pairs_df: pd.DataFrame, verbose: bool = True):
    """
    Given the pair-level DataFrame (with column D),
    keep non-tied pairs and compute:
        - n       : number of non-tied pairs
        - N_plus  : treated wins (D = 1)
        - N_minus : control wins (D = -1)
        - tau_hat : mean(D)
        - p0      : sign-test p-value under Γ = 1 (no hidden bias)

    Returns a dict with all these values plus the non_tied DataFrame.
    """
    # Keep non-tied pairs
    non_tied = pairs_df[pairs_df["D"] != 0].copy()

    n = non_tied.shape[0]                  # number of non-tied pairs
    N_plus = (non_tied["D"] == 1).sum()    # treated wins
    N_minus = (non_tied["D"] == -1).sum()  # control wins
    tau_hat = non_tied["D"].mean() if n > 0 else np.nan

    if n > 0:
        p0 = binomtest(N_plus, n, p=0.5, alternative="greater").pvalue
    else:
        p0 = np.nan

    if verbose:
        print("Number of non-tied pairs n       =", n)
        print("Treated wins N_plus              =", N_plus)
        print("Control wins N_minus             =", N_minus)
        print(f"Average difference tau_hat (D̄)  = {tau_hat:.3f}" if n > 0 else
              "Average difference tau_hat (D̄)  = nan")
        if n > 0:
            print(f"\nBaseline sign-test p-value (Γ = 1, no hidden bias): p0 = {p0:.4g}")
        else:
            print("\nNo non-tied pairs found (n = 0); sensitivity analysis is not meaningful.")

    return {
        "pairs_df": pairs_df,
        "non_tied": non_tied,
        "n": n,
        "N_plus": N_plus,
        "N_minus": N_minus,
        "tau_hat": tau_hat,
        "p0": p0,
    }

import pandas as pd
def rosenbaum_bounds(N_plus: int, n: int, gammas) -> pd.DataFrame:
    """
    Compute Rosenbaum upper-bound p-values for a sign test
    over a list of Γ values.

    Under hidden bias up to Γ, the worst-case probability that the
    treated unit "wins" in each pair is p_gamma = Γ / (1 + Γ).

    We then compute:
        p_upper(Γ) = P[ Binomial(n, p_gamma) >= N_plus ].

    Parameters
    ----------
    N_plus : int
        Number of non-tied matched pairs where the treated unit wins (D = 1).
    n : int
        Total number of non-tied pairs (D != 0).
    gammas : iterable of float
        Values of Γ to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - "Gamma"
            - "p_upper"
    """
    import numpy as np
    from scipy.stats import binom
    import pandas as pd
    from scipy.stats import binomtest

    rows = []

    for gamma in gammas:
        # Worst-case treated-win probability under hidden bias Γ
        p_gamma = gamma / (1.0 + gamma)

        # Binomial upper tail: P(X >= N_plus)
        # binom.sf(k-1, n, p) = P(X >= k)
        p_upper = binom.sf(N_plus - 1, n, p_gamma)

        rows.append({
            "Gamma": gamma,
            "p_upper": p_upper
        })

    return pd.DataFrame(rows)


def run_sensitivity_analysis(
    N_plus: int,
    n: int,
    alpha: float = 0.05,
    gamma_grid=None,
    verbose: bool = True,
):
    """
    Convenience wrapper to run Rosenbaum sensitivity analysis on a
    predefined Γ grid and find the sensitivity threshold Γ*.

    Parameters
    ----------
    N_plus : int
        Number of treated wins among non-tied pairs.
    n : int
        Number of non-tied pairs.
    alpha : float, default 0.05
        Significance level used to define Γ*.
    gamma_grid : array-like or None
        Γ values to evaluate. If None, uses a default grid:
        [1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0].
    verbose : bool, default True
        If True, prints the table and Γ*.

    Returns
    -------
    bounds_df : pd.DataFrame
        Output of `rosenbaum_bounds`, with columns ["Gamma", "p_upper"].
    gamma_star : float or None
        Smallest Γ such that p_upper > alpha, or None if no such Γ
        exists on the grid.
    """
    import numpy as np
    from scipy.stats import binom
    import pandas as pd
    from scipy.stats import binomtest

    if n <= 0:
        if verbose:
            print("No non-tied pairs (n = 0); sensitivity analysis not meaningful.")
        return pd.DataFrame(), None

    if gamma_grid is None:
        gamma_grid = np.array([1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0])
    else:
        gamma_grid = np.array(gamma_grid, dtype=float)

    bounds_df = rosenbaum_bounds(N_plus=N_plus, n=n, gammas=gamma_grid)

    # Find first Γ where p_upper > alpha
    above_alpha = bounds_df[bounds_df["p_upper"] > alpha]
    if not above_alpha.empty:
        gamma_star = float(above_alpha.iloc[0]["Gamma"])
    else:
        gamma_star = None

    if verbose:
        print("Rosenbaum bounds:")
        print(bounds_df)
        if gamma_star is not None:
            print(f"\nSensitivity threshold Γ* (p_upper > {alpha}): Γ* ≈ {gamma_star}")
        else:
            print(f"\nEffect remains significant (p_upper ≤ {alpha}) "
                  f"for all Γ up to {gamma_grid.max()}.")

    return bounds_df, gamma_star


def build_pairs_from_matched(matched_with_Y: pd.DataFrame) -> pd.DataFrame:
    """
    From the matched dataset (with one treated and one control row
    per matched_pair_id), construct a pair-level table with:
        - matched_pair_id
        - Y_treated, Y_control
        - D = Y_treated - Y_control ∈ {-1,0,1}
    """
    rows = []

    for pid, group in matched_with_Y.groupby("matched_pair_id"):
        # Grab treated and control rows
        treated_rows = group[group["treated"] == 1]
        control_rows = group[group["treated"] == 0]

        # We expect exactly one treated and one control per matched_pair_id
        if len(treated_rows) != 1 or len(control_rows) != 1:
            # If this happens, we skip or could handle separately
            continue

        yt = int(treated_rows["Y"].iloc[0])
        yc = int(control_rows["Y"].iloc[0])

        rows.append({
            "matched_pair_id": pid,
            "Y_treated": yt,
            "Y_control": yc,
            "D": yt - yc
        })

    pairs_df = pd.DataFrame(rows)
    return pairs_df

