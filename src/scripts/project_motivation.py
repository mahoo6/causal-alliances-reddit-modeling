import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from IPython.display import display



def build_monthly_counts(edges: pd.DataFrame):
    pm = (edges.groupby(["month","SOURCE_SUBREDDIT","TARGET_SUBREDDIT","LINK_SENTIMENT"])
                .size().rename("n").reset_index()
                .pivot(index=["month","SOURCE_SUBREDDIT","TARGET_SUBREDDIT"], columns="LINK_SENTIMENT", values="n")
                .fillna(0).rename(columns={-1:"neg", +1:"pos"}).reset_index())
    if "pos" not in pm: pm["pos"] = 0
    if "neg" not in pm: pm["neg"] = 0

    sm = (pm.groupby(["month","SOURCE_SUBREDDIT"])[["pos","neg"]]
             .sum().reset_index()
             .rename(columns={"SOURCE_SUBREDDIT":"subreddit","pos":"out_pos","neg":"out_neg"}))
    sm["out_total"] = sm["out_pos"] + sm["out_neg"]
    return pm, sm

def negout_per_month_summary(value_col: str, sub_month: pd.DataFrame):
     # Negative out-degree: for each subreddit and month, total # of negative links sent.
    neg_out = sub_month.rename(columns={"subreddit": "SOURCE_SUBREDDIT", "out_neg": "neg_out"})[["month","SOURCE_SUBREDDIT","neg_out"]]
    g = neg_out.groupby("month")[value_col]
    return pd.DataFrame({
        "median": g.median(),
        "p25":    g.quantile(0.25),
        "p75":    g.quantile(0.75),
        "p90":    g.quantile(0.90),
        "p99":    g.quantile(0.99),
        "mean":   g.mean(),
        "count":  g.size(),
        "min":   g.min(),
        "max":    g.max()
    }).reset_index()

def negin_per_month_summary(value_col: str, pair_month: pd.DataFrame):
    neg_in = (pair_month.groupby(["month","TARGET_SUBREDDIT"])["neg"]
            .sum().reset_index()
            .rename(columns={"TARGET_SUBREDDIT": "subreddit", "neg": "neg_in"}))
    g = neg_in.groupby("month")[value_col]
    return pd.DataFrame({
        "median": g.median(),
        "p25":    g.quantile(0.25),
        "p75":    g.quantile(0.75),
        "p90":    g.quantile(0.90),
        "p99":    g.quantile(0.99),
        "mean":   g.mean(),
        "count":  g.size(),
        "min":   g.min(),
        "max":    g.max()
    }).reset_index()

def boxplot_from_summary(df, title):
    # Ensure months are sorted in chronological order
    df = df.sort_values("month")
    months = df["month"].tolist()

    # Build list of boxplot dicts from summary stats
    box_data = []
    for _, row in df.iterrows():
        box_data.append({
            'whislo': row['min'],      # bottom whisker
            'q1': row['p25'],          # first quartile
            'med': row['median'],      # median
            'q3': row['p75'],          # third quartile
            'whishi': row['max'],      # top whisker
            'fliers': []               # no fliers (since we’re summarizing)
        })

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bxp(box_data, showfliers=False)
    ax.set_xticks(np.arange(1, len(months)+1))
    ax.set_xticklabels(months, rotation=45, ha='right')
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("# Negative Links")
    plt.tight_layout()
    plt.show()

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

    Returns columns: ['month','n_TARGET_SUBREDDITs_multi','n_TARGET_SUBREDDITs_total','pct_multi', 'month_dt']
    """
    # 1) keep only negative (attack) links
    neg_edges = pair_month.loc[pair_month[neg_col] > 0, [month_col, SOURCE_SUBREDDIT_col, TARGET_SUBREDDIT_col]].copy()

    # 2) for each (month, TARGET_SUBREDDIT), count distinct attackers
    monthly_attackers = (
        neg_edges.groupby([month_col, TARGET_SUBREDDIT_col])[SOURCE_SUBREDDIT_col]
        .nunique()
        .reset_index(name="n_attackers")
    )

    # 3) per-month totals and multi-attacked counts
    totals = (monthly_attackers
              .groupby(month_col)
              .size()
              .reset_index(name="n_TARGET_SUBREDDITs_total"))

    multi = (monthly_attackers
             .assign(is_multi=lambda df: df["n_attackers"] >= threshold)
             .groupby(month_col)["is_multi"]
             .sum()
             .reset_index(name="n_TARGET_SUBREDDITs_multi"))

    summary = totals.merge(multi, on=month_col, how="left").fillna(0)
    summary["pct_multi"] = summary["n_TARGET_SUBREDDITs_multi"] / summary["n_TARGET_SUBREDDITs_total"]

    # 4) sort months chronologically
    try:
        summary["month_dt"] = pd.PeriodIndex(summary[month_col].astype(str), freq="M").to_timestamp()
    except Exception:
        # fallback: try datetime parsing; if it fails, keep as-is
        summary["month_dt"] = pd.to_datetime(summary[month_col], errors="coerce")
    summary = summary.sort_values("month_dt")

    # 5) plot
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(
        data=summary,
        x="month_dt",
        y="pct_multi",
        marker="o",
        linewidth=2,
        ax=ax
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title(f"Share of TARGET_SUBREDDITs per Month with ≥ {threshold} Distinct Attackers")
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent of TARGET_SUBREDDITs (co-attacked)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    return summary

def build_month_index_simple(pair_month: pd.DataFrame):
    # Create a simple numeric index for each unique month in your dataset
    months_sorted = sorted(pair_month["month"].unique())
    month_to_idx = {m: i + 1 for i, m in enumerate(months_sorted)}   # '2014-01' -> 1
    idx_to_month = {i + 1: m for i, m in enumerate(months_sorted)}   # 1 -> '2014-01'

    print("Total months:", len(month_to_idx))
    print("First few entries:")
    print({k: month_to_idx[k] for k in list(month_to_idx.keys())})

    return months_sorted, month_to_idx, idx_to_month



def build_target_attacks_panel(pair_month: pd.DataFrame,
                               months_sorted: list,
                               month_to_idx: dict) -> pd.DataFrame:

    # Keep only negative interactions
    neg_df = pair_month.loc[pair_month["neg"] > 0, ["month", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "neg"]].copy()

    # Clean month to string YYYY-MM (to match month_to_idx keys)
    neg_df["month"] = pd.to_datetime(neg_df["month"], errors="coerce").dt.to_period("M").astype(str)

    # Total number of months
    T = len(months_sorted)

    # Build a dict for fast lookup
    rows = {}

    # Loop through all records
    for _, row in neg_df.iterrows():
        m, A, C, n = row["month"], row["SOURCE_SUBREDDIT"], row["TARGET_SUBREDDIT"], row["neg"]
        if m not in month_to_idx:
            continue
        t_idx = month_to_idx[m] - 1  # convert to 0-based index
        key = (C, A)

        if key not in rows:
            rows[key] = np.zeros(T, dtype=np.int32)

        rows[key][t_idx] += int(n)

    # Convert to DataFrame
    data = []
    for (C, A), vals in rows.items():
        row = {"C": C, "A": A}
        for i in range(1, T + 1):
            row[f"month_{i}"] = int(vals[i - 1])
        data.append(row)

    panel = pd.DataFrame(data)
    return panel

def find_high_activity_pairs(TARGET_SUBREDDIT_attack_panel: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """
    Select rows where any monthly count exceeds n and display the first 200.
    Returns the filtered DataFrame.
    """
    # Select only the month columns (they all start with "month_")
    month_cols = [c for c in TARGET_SUBREDDIT_attack_panel.columns if c.startswith("month_")]

    # Filter rows where ANY of those columns has a value > n
    high_activity = TARGET_SUBREDDIT_attack_panel[TARGET_SUBREDDIT_attack_panel[month_cols].gt(n).any(axis=1)]

    # Display the first few matching rows
    try:
        from IPython.display import display
        display(high_activity.head(200))
    except Exception:
        print(high_activity.head(200).to_string(index=False))

    print(f"The number of attacker-TARGET_SUBREDDIT pairs (A → C) that ever had more than {n} negative posts in any single month is: {high_activity.shape[0]}")

    return high_activity

def compute_row_stats(TARGET_SUBREDDIT_attack_panel: pd.DataFrame,
                      month_prefix: str = "month_",
                      preview: int = 100) -> pd.DataFrame:
    """
    From a wide panel with monthly counts (columns starting with month_prefix),
    compute per-row summary stats and return a compact table with:
      ['C','A','sum_attacks','mean_attacks','max_attacks',
       'min_attacks','std_attacks','months_active']
    """
    # Select month columns
    month_cols = [c for c in TARGET_SUBREDDIT_attack_panel.columns if c.startswith(month_prefix)]
    if not month_cols:
        raise ValueError(f"No month columns found with prefix '{month_prefix}'.")

    # Create new summary columns
    row_stats = TARGET_SUBREDDIT_attack_panel.copy()
    row_stats["sum_attacks"]   = row_stats[month_cols].sum(axis=1)
    row_stats["mean_attacks"]  = row_stats[month_cols].mean(axis=1)
    row_stats["max_attacks"]   = row_stats[month_cols].max(axis=1)
    row_stats["min_attacks"]   = row_stats[month_cols].min(axis=1)
    row_stats["std_attacks"]   = row_stats[month_cols].std(axis=1)
    row_stats["months_active"] = (row_stats[month_cols] > 0).sum(axis=1)

    # Keep requested columns (assumes 'C' and 'A' exist)
    row_stats = row_stats[[
        "C", "A", "sum_attacks", "mean_attacks", "max_attacks",
        "min_attacks", "std_attacks", "months_active"
    ]]

    # Show a few examples
    if preview and preview > 0:
        try:
            from IPython.display import display
            display(row_stats.head(preview))
        except Exception:
            print(row_stats.head(preview).to_string(index=False))

    return row_stats

def build_pair_event_panel(target_attack_panel, months_sorted, month_to_idx):

    from itertools import combinations

    month_cols = [c for c in target_attack_panel.columns if c.startswith("month_")]
    T = len(months_sorted)
    records = []

    # Loop through each target subreddit
    for C, grp in target_attack_panel.groupby("C"):
        attackers = grp.set_index("A")[month_cols].astype(int)

        # Loop through each month
        for t_idx, mcol in enumerate(month_cols, start=1):
            active = attackers[attackers[mcol] > 0][mcol]
            if len(active) < 2:
                continue  # skip months with < 2 attackers

            atk_dict = active.to_dict()
            # Generate all unordered attacker pairs
            for A, B in combinations(sorted(atk_dict.keys()), 2):
                records.append({
                    "C": C,
                    "A": A,
                    "B": B,
                    "month_idx": t_idx,
                    "pair_val": (atk_dict[A], atk_dict[B])
                })

    # If no co-attacks found, return empty panel
    if not records:
        return pd.DataFrame(columns=["C", "A", "B"] + [f"month_{i}" for i in range(1, T+1)])

    # Build DataFrame
    df = pd.DataFrame(records)

    # Pivot to wide format
    panel = df.pivot_table(index=["C", "A", "B"],
                           columns="month_idx",
                           values="pair_val",
                           aggfunc="first")

    # Flatten MultiIndex columns safely
    panel.columns = [f"month_{int(c)}" for c in panel.columns]
    panel = panel.reset_index()

    # Ensure all months 1..T exist (fill missing with NaN)
    for i in range(1, T + 1):
        col_name = f"month_{i}"
        if col_name not in panel.columns:
            panel[col_name] = np.nan

    # Reorder columns numerically
    ordered_cols = ["C", "A", "B"] + [f"month_{i}" for i in range(1, T + 1)]
    panel = panel[ordered_cols]

   
    panel = panel.sort_values(["C", "A", "B"]).reset_index(drop=True)
    return panel

def run_snippet(row_stats, mth, max_att):
    display(row_stats[(row_stats['months_active'] == mth) & (row_stats['max_attacks'] > max_att)])
    print(
        f"The number of attacker-target pairs where the source performed more than {max_att} attacks within {mth} active month is: "
        f"{row_stats[(row_stats['months_active'] == mth) & (row_stats['max_attacks'] > max_att)].shape[0]}"
    )

def run_strong_attack_check(pair_event_panel, x):
    # Get only the month columns
    month_cols = [c for c in pair_event_panel.columns if c.startswith("month_")]
    # Define a function to check if any month tuple exceeds x attacks in both value
    def has_strong_attack(row):
        for c in month_cols:
            val = row[c]
            if isinstance(val, tuple):
                if val[0] > x and val[1] > x:
                    return True
        return False

    # Apply the function row-wise
    mask = pair_event_panel.apply(has_strong_attack, axis=1)

    # Filter and display
    strong_pairs = pair_event_panel[mask]
    display(strong_pairs.head(10))
    print(f"Total pairs (A,B) with >{x} attacks each in any month: {len(strong_pairs)}")

def build_pair_summary(pair_event_panel):
    # Identify month columns
    month_cols = [c for c in pair_event_panel.columns if c.startswith("month_")]

    # Function to compute stats for each row
    def coattack_stats(row):
        start = np.nan
        active = 0
        for i, col in enumerate(month_cols, start=1):
            val = row[col]
            if isinstance(val, tuple):
                if np.isnan(start):
                    start = i        # first co-attack month
                active += 1
        return pd.Series({"start_month": start, "active_months": active})

    # Apply the function to each row
    pair_summary = pair_event_panel[["C", "A", "B"]].copy()
    pair_summary[["start_month", "active_months"]] = pair_event_panel.apply(coattack_stats, axis=1)
    return pair_summary

def run_active_months_check(pair_summary, mths):
    mask = pair_summary['active_months'] > mths
    display(pair_summary[pair_summary["active_months"] > mths])
    print(
        f"The number of pairs (A,B)-C where co-attackers were active at the same time for more than {mths} months is: "
        f"{pair_summary.loc[mask].shape[0]}"
    )


def build_pair_friendship_panel(pair_event_panel, pair_month, months_sorted, month_to_idx):
    
    # Extract unique (A,B) pairs (A<B to keep ordering consistent)
    pairs = pair_event_panel[["A", "B"]].drop_duplicates().sort_values(["A", "B"]).reset_index(drop=True)

    # Keep only positive links
    pos_df = pair_month.loc[pair_month["pos"] > 0, ["month", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "pos"]].copy()

    # Normalize month string format
    pos_df["month"] = pos_df["month"].astype(str)

    # Map month to index number (1..T)
    pos_df["month_idx"] = pos_df["month"].map(month_to_idx)
    T = len(months_sorted)

    # Create a lookup dict for quick retrieval: (A,B,month_idx) -> count
    lookup = {(r.SOURCE_SUBREDDIT, r.TARGET_SUBREDDIT, r.month_idx): r.pos for r in pos_df.itertuples(index=False)}

    records = []
    for row in pairs.itertuples(index=False):
        A, B = row.A, row.B
        rec = {"A": A, "B": B}
        for i in range(1, T + 1):
            x = lookup.get((A, B, i), 0)
            y = lookup.get((B, A, i), 0)
            rec[f"month_{i}"] = (x, y) if (x > 0 or y > 0) else np.nan
        records.append(rec)

    panel = pd.DataFrame(records)
    return panel.sort_values(["A", "B"]).reset_index(drop=True)

def get_active_pairs(friendship_panel):
    # Identify all month columns
    month_cols = [c for c in friendship_panel.columns if c.startswith("month_")]

    # Keep rows where ANY month column is not NaN
    active_pairs = friendship_panel[friendship_panel[month_cols].notna().any(axis=1)]

    # Display first few results
    display(active_pairs.head(20))
    print("Number of active (A,B) pairs with at least one positive link:", len(active_pairs))
    return active_pairs

def build_friendship_stat(friendship_panel):
    # Identify all month columns
    month_cols = [c for c in friendship_panel.columns if c.startswith("month_")]

    def friendship_stats(row):
        start = 0          # default = 0 (no positive links)
        active = 0
        for i, col in enumerate(month_cols, start=1):
            val = row[col]
            if isinstance(val, tuple):   # if at least one positive link that month
                if start == 0:           # first active month
                    start = i
                active += 1              # count all active months
        return pd.Series({"start_month": start, "active_months": active})

    # Build the summary DataFrame
    friendship_stat = friendship_panel[["A", "B"]].copy()
    friendship_stat[["start_month", "active_months"]] = friendship_panel.apply(friendship_stats, axis=1)

    display(friendship_stat.head(10))
    print(friendship_stat.shape)
    return friendship_stat

def build_conflict_friendship_comparison(pair_event_panel: pd.DataFrame,
                                         pair_summary: pd.DataFrame,
                                         friendship_panel: pd.DataFrame,
                                         friendship_stat: pd.DataFrame,
                                         months_sorted: list) -> pd.DataFrame:

    month_cols = [c for c in friendship_panel.columns if c.startswith("month_")]
    T = len(months_sorted)

    # Build fast lookup for friendship totals per month: (A,B) -> np.array length T with totals (x+y) or 0
    def to_series_of_totals(row):
        totals = np.zeros(T, dtype=np.int32)
        for i, col in enumerate(month_cols, start=1):
            val = row[col]
            if isinstance(val, tuple):
                totals[i-1] = int(val[0]) + int(val[1])
            else:
                totals[i-1] = 0
        return totals

    fp = friendship_panel.set_index(["A", "B"])
    friendship_lookup = {idx: to_series_of_totals(fp.loc[idx]) for idx in fp.index}

    # Conflict and friendship start lookups
    conflict_start_map = pair_summary.set_index(["C", "A", "B"])["start_month"].to_dict()
    friendship_start_map = friendship_stat.set_index(["A", "B"])["start_month"].to_dict()

    
    out_rows = []
    for r in pair_event_panel[["C", "A", "B"]].itertuples(index=False):
        C, A, B = r.C, r.A, r.B

        conflict_start = conflict_start_map.get((C, A, B), np.nan)
        friendship_start = friendship_start_map.get((A, B), 0)

        totals = friendship_lookup.get((A, B), np.zeros(T, dtype=np.int32))

        # compute pre/post sums
        if pd.isna(conflict_start) or conflict_start < 1:
            pre_sum = 0
            post_sum = int(totals.sum())
        else:
            cs = int(conflict_start)
            pre_sum = int(totals[:max(0, cs - 1)].sum())
            post_sum = int(totals[cs - 1:].sum())

        friendship_observed = (friendship_start > 0)
        new_friendship = (pre_sum == 0 and friendship_observed)

        out_rows.append({
            "C": C,
            "A": A,
            "B": B,
            "conflict_start": conflict_start,
            "friendship_start": friendship_start,
            "friendship_observed": friendship_observed,
            "pre_sum": pre_sum,
            "post_sum": post_sum,
            "friendship_evolution": post_sum - pre_sum,
            "new_friendship": new_friendship
        })

    result = pd.DataFrame(out_rows).sort_values(["C", "A", "B"]).reset_index(drop=True)
    return result

def plot_friendship_outcomes_pie(comparison):
    import matplotlib.pyplot as plt
    import numpy as np

    # --- masks / counts (same as before) ---
    new_mask    = comparison['new_friendship']
    never_mask  = ~comparison['friendship_observed']
    stayed_mask = comparison['friendship_observed'] & ~comparison['new_friendship']

    n = len(comparison)
    counts = [new_mask.sum(), never_mask.sum(), stayed_mask.sum()]
    labels_defs = [
        "New friends (no prior friendship; became friends after co-attack)",
        "Never friends (no friendship before or after)",
        "Stayed friends (were friends before; remained friends)"
    ]
    sizes = [c / n for c in counts]

    # --- plot compact, with legend outside and minimal whitespace ---
    fig, ax = plt.subplots(figsize=(9, 5))  # wider, less vertical white
    wedges, texts, autotexts = ax.pie(
        sizes,
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        counterclock=False
    )
    ax.axis("equal")

    # Centered title, smaller pad
    ax.set_title("Co-attacking Pair Friendship Outcomes", fontsize=18, pad=8, loc="center")

    # Give legend its own narrow right gutter
    ax.set_position([0.06, 0.12, 0.62, 0.76])  # [left, bottom, width, height]

    legend_entries = [f"{lab} — {cnt} ({cnt/n:.1%})" for lab, cnt in zip(labels_defs, counts)]
    ax.legend(
        wedges, legend_entries,
        title="Outcome (count, % of pairs)",
        title_fontsize=10, fontsize=9,
        loc="center left", bbox_to_anchor=(1.00, 0.5),
        frameon=False, borderaxespad=0
    )

    # When saving, this trims any remaining margins
    # plt.savefig("friendship_outcomes_pie.png", bbox_inches="tight", dpi=200)
    plt.show()

def plot_new_friendship_timelines(comparison, idx_to_month, C_SEL="leagueoflegends", TOPN=25):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from IPython.display import display

    # --- Parameters ---
    # C_SEL = "leagueoflegends"   # e.g. "politics"; if None, auto-picks target with most new_friendships
    # TOPN  = 25     # number of pairs (A,B) to display

    # --- Step 1: Filter to pairs that became new friends ---
    df = comparison.loc[comparison["new_friendship"] == True].copy()

    # Auto-pick the target with most new friendships if not specified
    if C_SEL is None:
        C_SEL = df["C"].value_counts().idxmax()

    dfC = df.loc[df["C"] == C_SEL].copy()
    if dfC.empty:
        raise ValueError(f"No new_friendship rows found for target C = {C_SEL!r}")

    # Compute time lag (friendship_start - conflict_start)
    dfC["delta"] = dfC["friendship_start"] - dfC["conflict_start"]
    dfC = dfC.loc[dfC["delta"] > 0].sort_values("delta").head(TOPN)

    # --- Step 2: Map indices to real months ---
    dfC["conflict_month"] = dfC["conflict_start"].map(idx_to_month)
    dfC["friendship_month"] = dfC["friendship_start"].map(idx_to_month)

    # --- Step 3: Plot timelines ---
    plt.figure(figsize=(10, 0.4 * len(dfC) + 2))

    y_positions = np.arange(len(dfC))
    for i, row in enumerate(dfC.itertuples(index=False)):
        plt.hlines(y=i,
                   xmin=row.conflict_start, xmax=row.friendship_start,
                   linewidth=2)
        plt.plot(row.conflict_start, i, 'o', color='red')     # co-attack
        plt.plot(row.friendship_start, i, 'o', color='green') # friendship

    # Replace numeric month indices with real month labels
    xmin, xmax = int(dfC["conflict_start"].min()), int(dfC["friendship_start"].max())
    step = max(1, (xmax - xmin) // 10)
    xticks_idx = list(range(xmin, xmax + 1, step))
    xticks_labels = [idx_to_month.get(i, "") for i in xticks_idx]

    plt.xticks(xticks_idx, xticks_labels, rotation=45, ha="right")
    plt.yticks(y_positions, [f"{r.A} ↔ {r.B}" for r in dfC.itertuples(index=False)])
    plt.xlabel("Month")
    plt.ylabel("Attacker Pair (A,B)")
    plt.title(f"When Co-Attackers Became Friends — Target C = {C_SEL}")
    plt.grid(axis='x', linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.show()

    # --- Optional: summary table ---
    display(dfC[["A", "B", "C", "conflict_month", "friendship_month", "delta"]].head(TOPN))
    return dfC

def build_pair_friendship_panel_strict(pair_event_panel, pair_month, months_sorted, month_to_idx):
    
    # Extract unique (A,B) pairs (A<B to keep ordering consistent)
    pairs = pair_event_panel[["A", "B"]].drop_duplicates().sort_values(["A", "B"]).reset_index(drop=True)

    # Keep only positive links
    pos_df = pair_month.loc[pair_month["pos"] > 0, ["month", "SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "pos"]].copy()

    # Normalize month string format
    pos_df["month"] = pos_df["month"].astype(str)

    # Map month to index number (1..T)
    pos_df["month_idx"] = pos_df["month"].map(month_to_idx)
    T = len(months_sorted)

    # Create a lookup dict for quick retrieval: (A,B,month_idx) -> count
    lookup = {(r.SOURCE_SUBREDDIT, r.TARGET_SUBREDDIT, r.month_idx): r.pos for r in pos_df.itertuples(index=False)}

    records = []
    for row in pairs.itertuples(index=False):
        A, B = row.A, row.B
        rec = {"A": A, "B": B}
        for i in range(1, T + 1):
            x = lookup.get((A, B, i), 0)
            y = lookup.get((B, A, i), 0)
            rec[f"month_{i}"] = (x, y) if (x > 0 and y > 0) else np.nan
        records.append(rec)

    panel = pd.DataFrame(records)
    return panel.sort_values(["A", "B"]).reset_index(drop=True)
