from src.scripts.causal_analysis_plots import (
    plot_pct_multi_attacked_TARGET_SUBREDDITs, plot_friendship_score_distribution, plot_attack_count_distribution_from_pair_event_panel, plot_conflict_friendship_timelines_basic, plot_causal_dag,
    plot_new_friendship_timelines,
    plot_friendship_outcomes_pie,
    plot_confounders_separately,
    plot_pscore_overlap,
    plot_logit_coefficients,
    plot_roc_for_propensity,
    plot_match_distance,
    plot_pscore_pairs,
    plot_confounders_separately_after_matching,
    plot_pscore_distribution_after_matching,
    plot_treatment_outcome_matrix,
    plot_bootstrap_att,
    plot_sensitivity_curve
)

from src.scripts.causal_analysis import (
    add_month_index,
    build_monthly_counts,
    build_monthly_unordered_pair_scores,
    learn_friend_enemy_thresholds,
    classify_monthly_relationship,
    build_pair_event_panel_from_df,
    build_enemy_status_lookup,
    filter_pair_event_panel_by_enemy_status,
    build_pair_summary,
    build_monthly_score_lookup,
    build_pair_friendship_score_panel_all_months,
    build_friendship_stat_from_score_panel,
    build_conflict_friendship_comparison_score_based,
    build_treated_pairs_from_comparison,
    build_global_friendship_stat_score_based,
    add_pair_activity_pre,
    add_pair_aggressiveness_pre,
    add_pair_similarity,
    add_preconflict_hostility,
    add_all_confounders,
    build_ps_dataset,
    fit_propensity_score_model,
    compute_propensity_scores,
    apply_common_support,
    nearest_neighbor_match,
    run_matching,
    add_outcome_to_matched,
    att_pairwise,
    bootstrap_att,
    build_friend_lookup_all,
    build_control_pairs,
    build_pairs_from_matched, 
    basic_sign_test_stats, 
    run_sensitivity_analysis
    )

import numpy as np
from scipy.stats import binom
import pandas as pd
from scipy.stats import binomtest

def causal_analysis_with_hidden(df_monthly, emb_df):
    df_monthly, months_sorted, month_to_idx, idx_to_month = add_month_index(df_monthly)
    pair_month, sub_month = build_monthly_counts(df_monthly)

    pair_monthly_scores = build_monthly_unordered_pair_scores(df_monthly)

    ENEMY_THRESHOLD, FRIEND_THRESHOLD, centers = learn_friend_enemy_thresholds(pair_monthly_scores)


    pair_monthly_scores["status"] = pair_monthly_scores["Friendship_Score"].apply(
        classify_monthly_relationship,
        enemy_threshold=ENEMY_THRESHOLD,
        friend_threshold=FRIEND_THRESHOLD
    )

    T = len(months_sorted)
    pair_event_panel_all = build_pair_event_panel_from_df(df_monthly, T)


    enemy_lookup = build_enemy_status_lookup(pair_monthly_scores)

    pair_event_panel = filter_pair_event_panel_by_enemy_status(
        pair_event_panel_all,
        enemy_lookup
    )

    pair_summary = build_pair_summary(pair_event_panel)
    
    score_lookup = build_monthly_score_lookup(pair_monthly_scores, score_col="Friendship_Score")

    friendship_score_panel = build_pair_friendship_score_panel_all_months(
        pair_event_panel,
        months_sorted,
        score_lookup
    )

    friendship_stat = build_friendship_stat_from_score_panel(friendship_score_panel, FRIEND_THRESHOLD)


    comparison = build_conflict_friendship_comparison_score_based(
        pair_summary=pair_summary,
        friendship_stat=friendship_stat
    )

    treated_pairs = build_treated_pairs_from_comparison(comparison)

    friendship_stat_all = build_global_friendship_stat_score_based(pair_monthly_scores)
    friend_lookup_all = build_friend_lookup_all(friendship_stat_all)

    control_pairs = build_control_pairs(
        df_monthly=df_monthly,
        treated_pairs=treated_pairs,
        pair_event_panel_all=pair_event_panel_all,
        friendship_stat_all=friendship_stat_all,
        seed=42
    )

    treated_pairs_conf, control_pairs_conf = add_all_confounders(
        treated_pairs.copy(),
        control_pairs.copy(),
        sub_month,
        emb_df,
        pair_month
    )

    
    df_ps, X, y, confounders = build_ps_dataset(
        treated_pairs_conf,
        control_pairs_conf
    )

    scaler, logit, X_scaled = fit_propensity_score_model(X, y)

    df_ps = compute_propensity_scores(df_ps, scaler, logit, confounders)


    confounders = ["activity", "aggressiveness", "similarity", "hostility_pre"]

    matched_df = run_matching(df_ps)

    confounders = ["activity", "aggressiveness", "similarity", "hostility_pre"]
    
    matched_with_Y = add_outcome_to_matched(matched_df, friend_lookup_all)

    att, diffs = att_pairwise(matched_with_Y)

    results = bootstrap_att(diffs, n_boot=5000, ci=95, seed=42)

    plot_bootstrap_att(results)

    pairs_df = build_pairs_from_matched(matched_with_Y)
    stats = basic_sign_test_stats(pairs_df, verbose=True)

    n = stats["n"]
    N_plus = stats["N_plus"]

    bounds_df, gamma_star = run_sensitivity_analysis(
        N_plus=N_plus,
        n=n,
        alpha=0.05,
        # gamma_grid=None  # or pass a custom list if you like
        verbose=True
    )

    fig = plot_sensitivity_curve(bounds_df, gamma_star=gamma_star, alpha=0.05)

    return None
