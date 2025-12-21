import numpy as np
import pandas as pd
import datetime as dt
import os
import torch
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from math import sqrt
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from scipy.stats import mannwhitneyu
import requests
import zipfile
import io


####################################################################################
####################################################################################
# UPLOADING AND MERGING DATASETS 
####################################################################################
####################################################################################

def extract_mobilization_data(base_path):
    """
    Loads train/val/test pickle files from the Stanford web source, 
    extracts Post IDs and Mobilization Labels, and returns a clean DataFrame.
    """
    url = "http://snap.stanford.edu/conflict/conflict_data.zip"
    files = [
        'preprocessed_train_data.pkl',
        'preprocessed_val_data.pkl',
        'preprocessed_test_data.pkl'
    ]
    
    # Ensure local directory exists
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Check if files are missing locally
    missing_files = [f for f in files if not os.path.exists(os.path.join(base_path, f))]
    
    if missing_files:
        print(f"Downloading {len(missing_files)} missing files from Stanford...")
        response = requests.get(url)
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for member in z.namelist():
                # The files are inside a 'prediction/' folder in the ZIP
                # We check if the current file in the zip matches our needed filenames
                filename = os.path.basename(member)
                if filename in missing_files and member.startswith('prediction/'):
                    # We extract it directly into base_path, removing the 'prediction/' prefix
                    with z.open(member) as source, open(os.path.join(base_path, filename), "wb") as target:
                        target.write(source.read())
                    print(f"Extracted: {filename}")

    all_post_ids = []
    all_labels = []
    
    for filename in files:
        file_path = os.path.join(base_path, filename)
        
        try:
            # Load the raw pickle (List of batches)
            raw_data = pd.read_pickle(file_path)
            
            # normalize iterator whether it's a list or dataframe wrapper
            if isinstance(raw_data, pd.DataFrame):
                iterator = raw_data.itertuples(index=False)
                is_df = True
            else:
                iterator = raw_data
                is_df = False

            for batch in iterator:
                # Column 0: Post IDs (List of byte-strings)
                raw_ids = batch[0]
                if is_df and hasattr(raw_ids, 'tolist'): 
                    raw_ids = raw_ids.tolist()
                
                # Decode b'id' -> 'id'
                clean_ids = [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in raw_ids]
                
                # Column 6: Labels (Tensor of 0s and 1s)
                raw_labels = batch[6]
                if isinstance(raw_labels, torch.Tensor):
                    clean_labels = raw_labels.cpu().numpy().astype(int)
                else:
                    clean_labels = np.array(raw_labels).astype(int)
                
                all_post_ids.extend(clean_ids)
                all_labels.extend(clean_labels)
                
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Create cleaned dataframe
    df_mob = pd.DataFrame({
        'POST_ID': all_post_ids,
        'MOBILIZATION_LABEL': all_labels
    })
    
    # Remove duplicates
    df_mob = df_mob.drop_duplicates(subset=['POST_ID'])
    return df_mob


def build_df_hostility(df, emb_df):
    
    df_hostility = []

    df_hostility = df[["SOURCE_SUBREDDIT", 'TARGET_SUBREDDIT', 'POST_ID', 'TIMESTAMP', 'LINK_SENTIMENT']].copy()

    # build unordered pair
    a = df["SOURCE_SUBREDDIT"].astype(str)
    b = df["TARGET_SUBREDDIT"].astype(str)
    df_hostility["pair_a"] = np.where(a < b, a, b)
    df_hostility["pair_b"] = np.where(a < b, b, a)

    # Build normalized embedding matrix
    E = emb_df.set_index('subreddit').astype(float)
    E = E.div(np.linalg.norm(E.to_numpy(), axis=1, keepdims=True), axis=0)

    # Filter rows that have both embeddings
    link_embeddings_available = df_hostility[
        df_hostility['SOURCE_SUBREDDIT'].isin(E.index) &
        df_hostility['TARGET_SUBREDDIT'].isin(E.index)
    ].copy()

    # Cosine similarity for rows with both embeddings (internal)
    src = E.reindex(link_embeddings_available['SOURCE_SUBREDDIT']).to_numpy()
    tgt = E.reindex(link_embeddings_available['TARGET_SUBREDDIT']).to_numpy()
    cos = pd.Series(np.einsum('ij,ij->i', src, tgt), index=link_embeddings_available.index, name='cosine_sim')

    # Add back to the dataset (set Nan for missing values)
    df_hostility['cosine_sim'] = np.nan
    df_hostility.loc[cos.index, 'cosine_sim'] = cos

    return df_hostility



def merge_mobilization(df_main, df_mob):
    """
    Merges mobilization data into df_hostility (df_main).
    Handles ID mismatch (trailing 's') AND duplicate post entries.
    """
    df = df_main.copy()
    
    # If the ID ends with 's', remove it. Otherwise keep it as is.
    df['clean_join_id'] = df['POST_ID'].astype(str).apply(lambda x: x[:-1] if x.endswith('s') else x)

    original_mob_ones = df_mob[df_mob['MOBILIZATION_LABEL'] == 1]['POST_ID'].nunique()
    
    # Prepare the Mobilization Dataframe for Merge
    # We drop duplicates here to ensure the source is unique on POST_ID
    df_mob_unique = df_mob[['POST_ID', 'MOBILIZATION_LABEL']].drop_duplicates(subset='POST_ID')
    
    df = df.merge(
        df_mob_unique, 
        left_on='clean_join_id', 
        right_on='POST_ID', 
        how='left',
        suffixes=('', '_mob')
    )
    
    # Fill missing values (NaN becomes 0)
    df['MOBILIZATION_LABEL'] = df['MOBILIZATION_LABEL'].fillna(0).astype(int)
    
    df = df.drop(columns=['clean_join_id', 'POST_ID_mob'], errors='ignore')
    
    # Calculate stats
    merged_mob_unique_posts = df[df['MOBILIZATION_LABEL'] == 1]['POST_ID'].nunique()
    
    if original_mob_ones > 0:
        recovery_pct = (merged_mob_unique_posts / original_mob_ones) * 100
    else:
        recovery_pct = 0.0

    print(f"Unique 'Mobilization=1' posts in source: {original_mob_ones}")
    print(f"Unique 'Mobilization=1' posts mapped:    {merged_mob_unique_posts}")
    print(f"Recovery Rate (Unique Posts):            {recovery_pct:.2f}%")
    
    return df



def add_text_s_flip_to_hostility_df(df_hostility, df_source, adaptive_flips, selected_features):
    """
    Enriches the hostility dataframe with both:
    1. Selected Textual Features (from df_source)
    2. Reaction Scores (s_flip from adaptive_flips)
    
    Uses a strict composite key (Post + Source + Target + Time) to prevent ANY row duplication.
    """
    df_final = df_hostility.copy()
    
    merge_keys = ['POST_ID', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']

    # Add Textual Features
    # Drop duplicates just in case the source df has exact duplicate rows
    text_source = df_source[merge_keys + selected_features].drop_duplicates(subset=merge_keys)
    
    df_final = df_final.merge(
        text_source,
        on=merge_keys,
        how='left',
        validate='many_to_one'
    )

    # Add s_flip Score
    # Prepare flip data: Rename columns to match df_hostility
    flip_source = adaptive_flips[[
        'Origin_Post_ID', 
        'Origin_Source', 
        'Origin_Target', 
        'Timestamp1', 
        's_flip'
    ]].rename(columns={
        'Origin_Post_ID': 'POST_ID',
        'Origin_Source': 'SOURCE_SUBREDDIT',
        'Origin_Target': 'TARGET_SUBREDDIT',
        'Timestamp1': 'TIMESTAMP'
    })
    
    # Ensure strictly one score per unique interaction
    flip_source = flip_source.drop_duplicates(subset=merge_keys)
    
    df_final = df_final.merge(
        flip_source,
        on=merge_keys,
        how='left',
        validate='many_to_one'
    )
    
    # Fill missing s_flip (=0)
    df_final['s_flip'] = df_final['s_flip'].fillna(0.0)
        
    return df_final



def merge_hidden_hostility_flags(df_main, df_results):
    """
    Merges the 'Hidden Hostility' flags into the main dataframe.
    """
    df_final = df_main.copy()
    
    merge_cols = ['POST_ID', 'SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
    
    results_subset = df_results[merge_cols + ['potential_mislabeled']].copy()
    
    df_final = df_final.merge(
        results_subset,
        on=merge_cols,
        how='left',
        validate='one_to_one' 
    )
    
    # Create the binary flag
    df_final['is_hidden_hostility'] = df_final['potential_mislabeled'].fillna(0).astype(int)
    
    df_final = df_final.drop(columns=['potential_mislabeled'])
    
    print(f"Total Hidden Hostilities Identified: {df_final['is_hidden_hostility'].sum()}")
    
    return df_final











####################################################################################
####################################################################################
# BUILDING NEAREST FLIPS  
####################################################################################
####################################################################################


def compute_all_intervals(df):
    # Ensure sorted by timestamp
    df = df.sort_values("TIMESTAMP", kind="mergesort")

    interval_dict = {}

    for (pa, pb), group in df.groupby(["pair_a", "pair_b"], sort=False):
        times = group["TIMESTAMP"].sort_values().to_numpy()

        if len(times) > 1:
            deltas = np.diff(times)
            interval_dict[(pa, pb)] = deltas
        else:
            interval_dict[(pa, pb)] = []   # only one event → no intervals

    return interval_dict

def compute_median_intervals(interval_dict):
    median_intervals = {}

    for pair, deltas in interval_dict.items():
        if len(deltas) > 0:
            median_intervals[pair] = np.median(deltas)
        else:
            median_intervals[pair] = pd.Timedelta("NaT")  # no interval possible

    return median_intervals


def build_followups_nearest_flip(df_hostility, median_intervals):
    """
    For each origin event within each unordered pair {A,B}, find the nearest future
    event of opposite LINK_SENTIMENT.
    
    Returns one row per origin that has a future flip:
      - Origin_Source, Origin_Target
      - Follow_Link_Direction ("same (src→tgt)" or "reverse (tgt→src)")
      - Timestamp1 (origin), Timestamp2 (flip), Time_Difference
      - Link_Sentiment1, Link_Sentiment2
      - pair_a, pair_b
      - Median_Interval (if provided)
      - s_flip = exp(-Δt / Median_Interval) when Median_Interval > 0
    """
    df = df_hostility.copy()
    med_map = median_intervals or {}

    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
    
    # Sort globally by timestamp
    df = df.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)

    out_rows = []

    # Group by unordered pairs
    for (pa, pb), g in df.groupby(["pair_a", "pair_b"], sort=False):
        g = g.sort_values("TIMESTAMP", kind="mergesort").reset_index(drop=True)

        ts   = g["TIMESTAMP"].to_numpy()
        sgn  = g["LINK_SENTIMENT"].to_numpy()
        src  = g["SOURCE_SUBREDDIT"].to_numpy()
        tgt  = g["TARGET_SUBREDDIT"].to_numpy()
        pids = g["POST_ID"].to_numpy()

        n = len(g)
        # Pre-index positions of +1 and -1
        idx_pos = np.where(sgn == +1)[0]
        idx_neg = np.where(sgn == -1)[0]
        ts_pos  = ts[idx_pos]
        ts_neg  = ts[idx_neg]

        # Median interval for this pair
        med_int = med_map.get((pa, pb), pd.NaT)

        for i in range(n):
            t0 = ts[i]
            s0 = sgn[i]

            # Find nearest future opposite sign
            if s0 == +1:
                j_rel = np.searchsorted(ts_neg, t0, side="right")
                if j_rel >= len(ts_neg): continue
                j = idx_neg[j_rel]
            else:
                j_rel = np.searchsorted(ts_pos, t0, side="right")
                if j_rel >= len(ts_pos): continue
                j = idx_pos[j_rel]

            # Build output row
            dt = ts[j] - t0
            if dt <= pd.Timedelta(0): continue

            row = {
                "Origin_Post_ID": pids[i],
                "Origin_Source": src[i],
                "Origin_Target": tgt[i],
                "Follow_Link_Direction": "same (src→tgt)" if (src[j] == src[i] and tgt[j] == tgt[i]) else "reverse (tgt→src)",
                "Timestamp1": t0,
                "Timestamp2": ts[j],
                "Time_Difference": dt,
                "Link_Sentiment1": int(s0),
                "Link_Sentiment2": int(sgn[j]),
                "pair_a": pa,
                "pair_b": pb,
                "Median_Interval": med_int,
            }

            # s_flip calculation
            try:
                if (pd.notna(med_int)) and (med_int != pd.Timedelta(0)):
                    row["s_flip"] = float(np.exp(-dt / med_int))
            except Exception:
                pass

            out_rows.append(row)

    return pd.DataFrame(out_rows)












####################################################################################
####################################################################################
# TEXTUAL FEATURE SELECTION 
####################################################################################
####################################################################################



def _oof_standardize_by_group(x, groups, skf):
    """
    Out-of-fold standardization within group.
    x: 1d array (feature)
    groups: group labels aligned with x
    Returns oof-standardized feature values.
    """
    oof = np.zeros_like(x, dtype=float)
    for train, test in skf.split(x, np.zeros_like(x)): 
        # per-group mean/std fitted on Training, applied to Test 
        g_train = pd.Series(groups[train])
        g_test = pd.Series(groups[test])
        x_train = pd.Series(x[train], index=g_train.index)
        x_test = pd.Series(x[test], index=g_test.index)

        mu = x_train.groupby(g_train).transform("mean")
        sd = x_train.groupby(g_train).transform("std").replace(0, np.nan)

        # map group stats to TE rows
        mu_map = pd.Series(mu.values, index=g_train).groupby(level=0).first()
        sd_map = pd.Series(sd.values, index=g_train).groupby(level=0).first()

        mu_test = g_test.map(mu_map).astype(float)
        sd_test = g_test.map(sd_map).astype(float).replace(0, np.nan)

        z = (x_test.values - mu_test.values) / sd_test.values
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        oof[test] = z
    return oof

def _univariate_oof_auc_with_direction(x, y, groups, skf):
    """
    OOF standardize-by-group, then compute raw AUC and direction.
    Returns:
        auc_adj: max(auc_raw, 1-auc_raw) in [0.5,1]
        direction: +1 if auc_raw >= 0.5 else -1
        oof_z: OOF standardized values (for optional diagnostics)
    """
    oof_z = _oof_standardize_by_group(x, groups, skf)

    auc_raw = roc_auc_score(y, oof_z)
    direction = 1 if auc_raw >= 0.5 else -1
    auc_adj = auc_raw if auc_raw >= 0.5 else (1.0 - auc_raw)
    return auc_adj, direction, oof_z



def select_textual_features_robust(
    df_hostility: pd.DataFrame,
    *,
    top=0.10,             
    corr_threshold=0.80,   
    l1_C=0.80,           
    folds=5,                
    repeats=10,               
    min_sign_consistency=0.80,
    min_test_auc=0.55,       
):
    """
    Robust selection of textual features:
      1) Repeated OOF AUC -> median rank + low rank variability
      2) Directional stability across repeats
      3) Time split test AUC on holdout
      4) Correlation pruning
      5) L1 + bootstrap stability
    Returns:
      selected: list of final robust features
      summary:  DataFrame with AUC stats, ranks, sign consistency, time AUC, kept flags
    """

    # Prepare data
    y = (df_hostility['LINK_SENTIMENT'] == -1).astype(int).to_numpy()
    TEXT_COLS = [c for c in df_hostility.columns if c.startswith("LIWC_")] + [c for c in df_hostility.columns if c.startswith("VADER_")]
    groups = df_hostility['SOURCE_SUBREDDIT'].astype(str).to_numpy()

    # Ensure timestamp
    df_tmp = df_hostility.copy()
    df_tmp['TIMESTAMP'] = pd.to_datetime(df_tmp['TIMESTAMP'], errors="coerce")
    df_tmp = df_tmp.dropna(subset=['TIMESTAMP'])

    # Repeated-CV univariate AUCs + ranks + direction
    rng = np.random.RandomState(7)
    rank_mat = []
    auc_mat  = []
    dir_mat  = []

    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=7 + r)

        stats_r = []
        dirs_r  = []
        for c in TEXT_COLS:
            x = df_hostility[c].astype(float).to_numpy()
            auc_adj, direction, _ = _univariate_oof_auc_with_direction(x, y, groups, skf)
            stats_r.append((c, auc_adj))
            dirs_r.append((c, direction))

        stats_r = pd.DataFrame(stats_r, columns=["feature", f"auc_oof_r{r}"])
        dirs_r  = pd.DataFrame(dirs_r,  columns=["feature", f"dir_r{r}"])
        auc_mat.append(stats_r)
        dir_mat.append(dirs_r)

        # ranks for this repeat
        ranks_r = stats_r.copy()
        ranks_r[f"rank_r{r}"] = ranks_r[f"auc_oof_r{r}"].rank(ascending=False, method="average")
        rank_mat.append(ranks_r[["feature", f"rank_r{r}"]])

    # Combine repeats
    summary = pd.DataFrame({"feature": TEXT_COLS})
    for df_r in auc_mat:
        summary = summary.merge(df_r, on="feature", how="left")
    for df_r in rank_mat:
        summary = summary.merge(df_r, on="feature", how="left")
    for df_r in dir_mat:
        summary = summary.merge(df_r, on="feature", how="left")

    auc_cols  = [c for c in summary.columns if c.startswith("auc_oof_r")]
    rank_cols = [c for c in summary.columns if c.startswith("rank_r")]
    dir_cols  = [c for c in summary.columns if c.startswith("dir_r")]

    summary["auc_med"]  = summary[auc_cols].median(axis=1)
    summary["auc_iqr"]  = summary[auc_cols].quantile(0.75, axis=1) - summary[auc_cols].quantile(0.25, axis=1)
    summary["rank_med"] = summary[rank_cols].median(axis=1)
    summary["rank_iqr"] = summary[rank_cols].quantile(0.75, axis=1) - summary[rank_cols].quantile(0.25, axis=1)

    # sign consistency in [0,1], count how often direction == +1 (or == -1) and take max
    pos_freq = (summary[dir_cols] == 1).mean(axis=1)
    neg_freq = (summary[dir_cols] == -1).mean(axis=1)
    summary["sign_consistency"] = np.maximum(pos_freq, neg_freq)

    if 0 < top < 1:
        k = max(1, int(len(summary) * top))
    else:
        k = int(top) if top >= 1 else max(1, int(len(summary) * 0.10))

    cand = (
        summary.sort_values("rank_med", ascending=True)
               .query("rank_iqr <= 0.15 * rank_med")   
               .head(k)
    )
    cand = cand[cand["sign_consistency"] >= min_sign_consistency]

    kept_after_rank = cand["feature"].tolist()

    # Time-split generalization (train early 70%, test late 30%)
    df_sorted = df_tmp.sort_values('TIMESTAMP')
    cut = int(0.7 * len(df_sorted))
    early, late = df_sorted.iloc[:cut], df_sorted.iloc[cut:]
    y_train = (early['LINK_SENTIMENT'] == -1).astype(int).to_numpy()
    y_test  = (late['LINK_SENTIMENT']  == -1).astype(int).to_numpy()
    groups_train = early['SOURCE_SUBREDDIT'].astype(str).to_numpy()
    groups_test  = late['SOURCE_SUBREDDIT'].astype(str).to_numpy()

    test_auc_list = []
    for f in kept_after_rank:
        # Standardize by group using only TRAIN stats; apply to TEST
        x_tr = early[f].astype(float).to_numpy()
        x_te = late[f].astype(float).to_numpy()

        # fit per-group stats on train
        gtr = pd.Series(groups_train)
        mu = pd.Series(x_tr).groupby(gtr).transform("mean")
        sd = pd.Series(x_tr).groupby(gtr).transform("std").replace(0, np.nan)

        mu_map = pd.Series(mu.values, index=gtr).groupby(level=0).first()
        sd_map = pd.Series(sd.values, index=gtr).groupby(level=0).first()

        gte = pd.Series(groups_test)
        mu_te = gte.map(mu_map).astype(float)
        sd_te = gte.map(sd_map).astype(float).replace(0, np.nan)

        z_te = (x_te - mu_te.values) / sd_te.values
        z_te = np.nan_to_num(z_te, nan=0.0, posinf=0.0, neginf=0.0)

        auc_raw = roc_auc_score(y_test, z_te)
        auc_adj = auc_raw if auc_raw >= 0.5 else (1.0 - auc_raw)
        test_auc_list.append((f, auc_adj))

    time_auc_df = pd.DataFrame(test_auc_list, columns=["feature", "test_auc"])
    cand = cand.merge(time_auc_df, on="feature", how="left")
    cand = cand[cand["test_auc"] >= min_test_auc]

    kept_after_time = cand["feature"].tolist()

    # Correlation pruning on kept features
    Z = df_hostility[['SOURCE_SUBREDDIT'] + kept_after_time].copy()
    # per-source z-scoring
    for c in kept_after_time:
        mu = Z.groupby('SOURCE_SUBREDDIT')[c].transform("mean")
        sd = Z.groupby('SOURCE_SUBREDDIT')[c].transform("std").replace(0, np.nan)
        Z[c] = ((Z[c] - mu) / sd).fillna(0.0)

    C = Z[kept_after_time].corr().abs()
    upper = C.where(np.triu(np.ones(C.shape), k=1).astype(bool))
    to_drop = set()
    # Tie-break by better rank_med (smaller is better)
    order = cand.sort_values("rank_med", ascending=True)["feature"].tolist()

    for col in order:
        if col in to_drop: 
            continue
        partners = upper.index[upper[col] >= corr_threshold].tolist()
        for p in partners:
            if p in to_drop or p == col:
                continue
            # drop the one with worse rank_med
            r_col = float(summary.loc[summary.feature==col, "rank_med"])
            r_p   = float(summary.loc[summary.feature==p,   "rank_med"])
            drop = p if r_col <= r_p else col
            to_drop.add(drop)

    sel_cols = [c for c in order if c not in to_drop]

    if len(sel_cols) == 0:
        print("No features kept after correlation pruning; falling back to time-kept set.")
        sel_cols = kept_after_time

    # Final L1 fit + bootstrap stability
    X = Z[sel_cols].to_numpy()

    sc = StandardScaler().fit(X)
    Xn = sc.transform(X)

    lr_l1 = LogisticRegression(penalty="l1", solver="liblinear",
                               class_weight="balanced", C=l1_C, max_iter=2000)
    lr_l1.fit(Xn, y)
    coef = lr_l1.coef_.ravel()
    selected = [c for c,w in zip(sel_cols, coef) if abs(w) > 1e-6]

    # Bootstrap stability of selected set
    rng = np.random.RandomState(7)
    freq = pd.Series(0, index=sel_cols, dtype=float)
    B = 20
    for _ in range(B):
        idx = rng.choice(len(Xn), size=len(Xn), replace=True)
        lr = LogisticRegression(penalty="l1", solver="liblinear",
                                class_weight="balanced", C=l1_C, max_iter=2000)
        lr.fit(Xn[idx], y[idx])
        nz = [c for c,w in zip(sel_cols, lr.coef_.ravel()) if abs(w) > 1e-6]
        freq.loc[nz] += 1
    stability = (freq / B)
    stable = list(stability[stability >= 0.60].index)
    print("Stability-selected (>=60%):", stable)

    # Build final summary table
    out = summary[["feature","auc_med","auc_iqr","rank_med","rank_iqr","sign_consistency"]].copy()
    out = out.merge(time_auc_df, on="feature", how="left")
    out["kept_after_rank"] = out["feature"].isin(kept_after_rank)
    out["kept_after_time"] = out["feature"].isin(kept_after_time)
    out["kept_after_corr"] = out["feature"].isin(sel_cols)
    out["selected_L1"]     = out["feature"].isin(selected)
    out["stable_boot"]     = out["feature"].isin(stable)

    return selected, out.sort_values(["selected_L1","stable_boot","kept_after_corr","auc_med"], ascending=[False, False, False, False])











####################################################################################
####################################################################################
# ROBUST HIDDEN HOSTILITY ANALYSIS   
####################################################################################
####################################################################################


def flag_potential_mislabeled_robust(
    df_hostility: pd.DataFrame,
    *,
    selected_feats,
    target_precision=0.85,
    min_activity_per_pair=5,
    per_source_cap=0.02,
    very_high_cut=0.99,
    sflip_quantile=0.95,
    folds=5,
    random_state=42,
    s_flip_threshold,
) -> pd.DataFrame:
    """
    Robust PU Learning flagger that handles missing data (e.g., missing embeddings)
    and generates a 'flag_reason' column for inspection.
    """

    df = df_hostility.copy()

    # Feature Selection & Setup
    feats = [f for f in selected_feats if f in df.columns]
    feats.append("cosine_sim")
        
    if len(feats) == 0:
        raise ValueError("No feature columns found to train on.")

    # Imputation & Scaling
    X_raw = df[feats].astype(float)
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_raw)
    
    scaler = StandardScaler()
    Xn = scaler.fit_transform(X_imputed)
    
    y = (df['LINK_SENTIMENT'] == -1).astype(int).to_numpy()
    
    # OOF Scoring (p_s)
    clf = LogisticRegression(solver="lbfgs", class_weight="balanced", max_iter=2000, random_state=random_state)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    p_s = cross_val_predict(clf, Xn, y, cv=skf, method="predict_proba")[:, 1]
    df["p_hostile_oof"] = p_s

    # PU Correction (Elkan-Noto)
    eps = 1e-8
    c = max(p_s[y == 1].mean() if (y == 1).any() else 0.5, eps)
    p_true = np.clip(p_s / c, 0.0, 1.0)
    df["p_true_hostile"] = p_true

    # Threshold Selection
    is_unlabeled_now = (df['LINK_SENTIMENT'] != -1).to_numpy()
    
    def est_precision_at(t):
        mask = is_unlabeled_now & (p_true >= t)
        if mask.sum() == 0: return 0.0
        return p_true[mask].mean()

    candidates = np.linspace(0.50, 0.99, 100)
    precs = np.array([est_precision_at(t) for t in candidates])
    
    valid_indices = np.where(precs >= target_precision)[0]
    t_star = candidates[valid_indices[0]] if len(valid_indices) > 0 else 0.99
    df["threshold_p_true"] = t_star

    # Corroboration & Flagging
    if 's_flip' in df.columns:
        valid_flips = df['s_flip'].dropna()
        sflip_thr = s_flip_threshold if len(valid_flips) > 0 else 1.0
        # Fill NaNs with 0 for logic checks
        sflip_vals = df['s_flip'].fillna(0.0)
    else:
        sflip_thr = 1.0
        sflip_vals = pd.Series(0, index=df.index)

    df["sflip_req"] = sflip_thr

    is_very_high = (p_true >= very_high_cut)
    is_high_and_flipped = (p_true >= t_star) & (sflip_vals >= sflip_thr)
    
    core_flag = is_very_high | is_high_and_flipped
    
    # Safety Checks
    pair_counts = df.groupby(["pair_a", "pair_b"])["TIMESTAMP"].transform("count")
    active_pair = pair_counts >= min_activity_per_pair
    prelim_flag = core_flag & active_pair & is_unlabeled_now

    # Per-source Cap
    if per_source_cap is not None and per_source_cap > 0:
        
        final_flag = pd.Series(0, index=df.index)
        
        for src, g in df.groupby('SOURCE_SUBREDDIT', sort=False):
            
            g_flags = prelim_flag.loc[g.index]
            n_flagged = g_flags.sum()
            
            if n_flagged > 0:
                n_total = len(g)
                allowed = max(1, int(np.floor(per_source_cap * n_total)))
                
                if n_flagged <= allowed:
                    
                    final_flag.loc[g.index] = g_flags.astype(int)
                else:
                    
                    flagged_indices = g.loc[g_flags].sort_values("p_true_hostile", ascending=False).index
                    kept_indices = flagged_indices[:allowed]
                    final_flag.loc[kept_indices] = 1
    else:
        final_flag = prelim_flag.astype(int)

    # Reasons Generation
    reasons = []
    for i, row in df.iterrows():
        r_list = []
        if not is_unlabeled_now[i]: 
            # Original negative, skip reason
            reasons.append("")
            continue
            
        p_val = row["p_true_hostile"]
        
        # Check why it was flagged (or almost flagged)
        if p_val >= very_high_cut:
            r_list.append(f"p_true>={very_high_cut:.2f}")
        elif p_val >= t_star:
            r_list.append(f"p_true≥t({t_star:.2f})")
            if 's_flip' in df.columns:
                val = row.get('s_flip', 0)
                if pd.notna(val) and val >= sflip_thr:
                    r_list.append(f"s_flip≥{sflip_thr:.2f}")
                else:
                    r_list.append(f"s_flip_low")
        
        # Check safety caps
        if not active_pair.loc[i]:
            r_list.append("low_activity")
        
        if final_flag.loc[i] == 0 and prelim_flag.loc[i] == 1:
            r_list.append("capped_by_source")
            
        reasons.append(";".join(r_list))

    df["flag_reason"] = reasons
    df["potential_mislabeled"] = final_flag.astype(int)
    
    return df








####################################################################################
####################################################################################
# MOBILIZATION ANALYSIS  
####################################################################################
####################################################################################

def analyze_mobilization_patterns(df_hostility, adaptive_flips, s_flip_threshold):
    """
    Identifies 'Aftershock' (Fake Peace) and 'Trojan Trigger' (Fake Friend) links.
    """
    
    # Clean input dataframe
    cols_to_drop = ['is_mobilization_aftershock', 'is_trojan_trigger', 'is_suspect']
    df_clean = df_hostility.drop(columns=[c for c in cols_to_drop if c in df_hostility.columns], errors='ignore').copy()
    
    # Setup Lookup for Mobilization ONLY
    # Drop duplicates to ensure unique index for mapping
    mob_lookup = df_clean[['POST_ID', 'MOBILIZATION_LABEL']].drop_duplicates('POST_ID').set_index('POST_ID')['MOBILIZATION_LABEL'].to_dict()
    
    # Filter Flips
    suspects = adaptive_flips.copy()
    
    # Map Mobilization to the ORIGIN event
    suspects['Origin_Mob'] = suspects['Origin_Post_ID'].map(mob_lookup).fillna(0)
    suspects['Origin_Sent'] = suspects['Link_Sentiment1']
    
    # PATTERN A: Mobilization Aftershock (Fake Peace)
    mask_aftershock = (
        (suspects['Origin_Mob'] == 1) &             # Mobilization
        (suspects['Origin_Sent'] == -1) &           # Hostile
        (suspects['Link_Sentiment2'] == 1) &        # Reaction: Positive
        (suspects['s_flip'] >= s_flip_threshold)    # Fast
    )
    
    # PATTERN B: Trojan Trigger (Fake Friend) 
    mask_trojan = (
        (suspects['Origin_Mob'] == 1) &             # Mobilization
        (suspects['Origin_Sent'] == 1) &            # Positive (Trojan)
        (suspects['Link_Sentiment2'] == -1) &       # Reaction: Hostile
        (suspects['s_flip'] >= s_flip_threshold)    # Fast
    )
    
    # Construct Merge Keys
    
    # A: Flag Reaction
    flips_A = suspects[mask_aftershock].copy()
    flips_A['Reaction_Source'] = np.where(flips_A['Follow_Link_Direction'] == 'same (src→tgt)', flips_A['Origin_Source'], flips_A['Origin_Target'])
    flips_A['Reaction_Target'] = np.where(flips_A['Follow_Link_Direction'] == 'same (src→tgt)', flips_A['Origin_Target'], flips_A['Origin_Source'])
    
    keys_A = flips_A[['Reaction_Source', 'Reaction_Target', 'Timestamp2']].copy()
    keys_A.columns = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
    keys_A['is_mobilization_aftershock'] = 1
    
    # B: Flag Origin
    flips_B = suspects[mask_trojan].copy()
    keys_B = flips_B[['Origin_Source', 'Origin_Target', 'Timestamp1']].copy()
    keys_B.columns = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP']
    keys_B['is_trojan_trigger'] = 1
    
    # Merge
    keys_A = keys_A.drop_duplicates(subset=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'])
    keys_B = keys_B.drop_duplicates(subset=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'])
    
    df_analyzed = df_clean.merge(keys_A, on=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'], how='left')
    df_analyzed = df_analyzed.merge(keys_B, on=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'TIMESTAMP'], how='left')
    
    df_analyzed['is_mobilization_aftershock'] = df_analyzed['is_mobilization_aftershock'].fillna(0).astype(int)
    df_analyzed['is_trojan_trigger'] = df_analyzed['is_trojan_trigger'].fillna(0).astype(int)
    
    return df_analyzed




def verify_hidden_attacks(flagged_robust, adaptive_flips, s_flip_threshold, min_sample=50):
    """
    Statistically verifies if the text model assigns higher hostility scores to 
    'Fake Peace' (Aftershock) links compared to normal positive links.
    """
    
    # Run the Pattern Analysis
    df_relaxed = analyze_mobilization_patterns(flagged_robust, adaptive_flips, s_flip_threshold=s_flip_threshold)

    mob_count = df_relaxed['is_mobilization_aftershock'].sum()
    print(f"Aftershocks found: {mob_count}")

    if mob_count < min_sample:
        print(f"⚠️ Sample size too small (<{min_sample}). Try lowering s_flip_threshold")
        return None

    # Define the two groups
    # Group A: The "Fake Peace" links (Structurally hostile, Linguistically positive)
    group_gold = df_relaxed[
        (df_relaxed['LINK_SENTIMENT'] == 1) & 
        (df_relaxed['is_mobilization_aftershock'] == 1)
    ]['p_true_hostile']

    # Group B: Normal Positive links (Baseline)
    group_baseline = df_relaxed[
        (df_relaxed['LINK_SENTIMENT'] == 1) & 
        (df_relaxed['is_mobilization_aftershock'] == 0)
    ]['p_true_hostile']

    # Compare Means
    mean_gold = group_gold.mean()
    mean_base = group_baseline.mean()
    
    print(f"Mean Hostility Score (Fake Peace):   {mean_gold:.4f}")
    print(f"Mean Hostility Score (Real Peace):   {mean_base:.4f}")

    # Statistical Test (Mann-Whitney U)
    # H0: The two groups come from the same distribution.
    # H1: Fake Peace has higher hostility scores than Real Peace.
    stat, p = mannwhitneyu(group_gold, group_baseline, alternative='greater')
    
    print(f"Mann-Whitney P-value: {p:.5e}")
    
    if p < 0.05:
        print("SUCCESS: The model statistically detects hidden aggression in 'Fake Peace' links.")
    else:
        print("RESULT: No statistical difference found.")
        
    return {'p_value': p, 'mean_gold': mean_gold, 'mean_baseline': mean_base}



###########################################################################
# creating new df_monthly for the ennemy of my ennemy analysis
###########################################################################

def monthly_hidden(df_hostility, df_monthly) :
    df_monthly_hidden = df_monthly.copy()
    df_monthly_hidden['is_hidden'] = df_hostility['is_hidden_hostility']
    df_monthly_hidden.loc[df_monthly_hidden["is_hidden"] == 1, "LINK_SENTIMENT"] = -1
    df_monthly_hidden = df_monthly_hidden.drop(columns="is_hidden")
    return df_monthly_hidden
