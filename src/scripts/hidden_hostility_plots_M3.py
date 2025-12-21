import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

alt.data_transformers.disable_max_rows()

ORANGE = '#ff6a00'
GRAY   = '#cfcfcf'
BLACK  = '#111111'
GRID   = '#ececec'

def hist_with_quantile_sflip(
    s,
    q=0.95,
    bins=60,
    title_prefix="Distribution of s_flip",
    xlabel="s_flip"
):
    # Defining the new high-contrast palette
    MAIN_BLUE = "#0077b6"    # High-pop contrast for orange websites
    CYAN_POP = "#00f5d4"     # The "eye-catcher" for lines
    TEXT_COLOR = "#2b2d42"   # Clean, professional dark gray
    GRID_COLOR = "#e0e0e0"

    s = np.asarray(s.dropna() if hasattr(s, "dropna") else s)
    s = s[np.isfinite(s)]
    if s.size == 0:
        raise ValueError("Empty / non-finite series after cleaning.")

    qv = np.quantile(s, q)
    prop = (s > qv).mean()

    fig, ax = plt.subplots(figsize=(10, 4.6))

    # 1. Histogram: Changed to deep blue with white edges for clarity
    counts, edges, patches = ax.hist(
        s,
        bins=bins,
        color=MAIN_BLUE,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85
    )

    # 2. Tail Highlight: Using a very soft cyan tint
    ax.axvspan(qv, edges[-1], facecolor=CYAN_POP, alpha=0.1, zorder=0)

    # 3. Quantile Line: Using a vibrant Cyan/Teal to "cut through" the blue bars
    ax.axvline(qv, color=CYAN_POP, linestyle='--', linewidth=3, zorder=5)

    # Annotation 
    y_top = counts.max() if len(counts) else 1.0
    ax.text(
        qv, y_top * 0.98,
        f"q{int(q*100)} = {qv:.3f}\nP(x > q{int(q*100)}) = {prop:.1%}",
        ha="left", va="top",
        fontsize=11, color=TEXT_COLOR,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=MAIN_BLUE, lw=1.5, alpha=1.0)
    )

    # Styling Titles and Labels
    ax.set_title(f"{title_prefix} (q{int(q*100)} = {qv:.3f})", 
                 fontsize=15, color=TEXT_COLOR, pad=15, fontweight='bold')
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=12)
    ax.set_ylabel("Count", color=TEXT_COLOR, fontsize=12)

    # Grid + Layout cleanup
    ax.grid(axis='y', color=GRID_COLOR, linestyle='-', alpha=0.7)
    ax.set_axisbelow(True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(TEXT_COLOR)
        
    ax.tick_params(colors=TEXT_COLOR)
    ax.set_xlim(min(0.0, s.min()), max(1.0, s.max()))

    plt.tight_layout()
    plt.show()



# ==========================================
# 1. CORE RECALLABLE FUNCTION
# ==========================================
def _compute_and_plot(df, categories_to_include, title):
    """
    Internal function that takes the dataframe, computes statistics 
    from scratch, and returns the interactive Altair chart.
    """

    # A. CATEGORIZATION 
    temp_df = df.copy()
    
    # Base Categorization: Neutral Links vs Explicit Enemy
    temp_df['Category'] = np.where(
        temp_df['LINK_SENTIMENT'] == -1, 
        'Explicit Enemy', 
        'Neutral Links'
    )
    
    # Add 'Hidden Hostility' ONLY if the column exists in the dataframe
    if 'potential_mislabeled' in temp_df.columns:
        # Overwrite 'Neutral Links' with 'Hidden Hostility' where flagged
        temp_df.loc[temp_df['potential_mislabeled'] == 1, 'Category'] = 'Hidden Hostility'
    
    # Filter to strictly requested categories
    temp_df = temp_df[temp_df['Category'].isin(categories_to_include)]

    # MELTING & STATS COMPUTATION 
    features = ['LIWC_Negemo', 'VADER_neg', 'VADER_compound', 'LIWC_Anger', 'LIWC_Affect']
    
    # Safety check: Ensure features exist before melting
    available_features = [f for f in features if f in temp_df.columns]
    if not available_features:
        raise ValueError(f"None of the required features {features} are in the dataframe.")

    long_df = temp_df.melt(
        id_vars=['Category'], 
        value_vars=available_features, 
        var_name='Feature', 
        value_name='Score'
    ).dropna(subset=['Score'])

    # Calculate exact 5th/95th percentiles (Robust Zoom)
    stats = long_df.groupby(['Category', 'Feature'])['Score'].agg(
        [
            ('min',    lambda x: np.percentile(x, 5)),
            ('q1',     lambda x: np.percentile(x, 25)),
            ('median', lambda x: np.percentile(x, 50)),
            ('q3',     lambda x: np.percentile(x, 75)),
            ('max',    lambda x: np.percentile(x, 95))
        ]
    ).reset_index()

    # ALTAIR PLOTTING 
    base = alt.Chart(stats).encode(
        x=alt.X('Category', axis=None, sort=['Neutral Links', 'Hidden Hostility', 'Explicit Enemy'])
    ).properties(width=130, height=300)

    # Whiskers (Using a dark charcoal for better integration with gray UI)
    rule = base.mark_rule(color='#2b2d42').encode(
        y=alt.Y('min', title=None),
        y2='max'
    )

    # Box (Updated Palette for Orange/Gray Website)
    bar = base.mark_bar(size=40, stroke='#2b2d42').encode(
        y='q1',
        y2='q3',
        fill=alt.Color('Category', scale=alt.Scale(
            domain=['Neutral Links', 'Explicit Enemy', 'Hidden Hostility'],
            # Teal, Navy-Indigo, and Electric Purple
            range=['#2a9d8f', "#264653", "#7209b7"] 
        ), legend=alt.Legend(title="Relationship Type")),
        
        tooltip=[
            alt.Tooltip('Category', title='Group'),
            alt.Tooltip('Feature', title='Feature'),
            alt.Tooltip('max',    format='.3f', title='95th Percentile'),
            alt.Tooltip('q3',     format='.3f', title='Q3 (75th)'),
            alt.Tooltip('median', format='.3f', title='Median'),
            alt.Tooltip('q1',     format='.3f', title='Q1 (25th)'),
            alt.Tooltip('min',    format='.3f', title='5th Percentile')
        ]
    )

    # Median (White Tick for visibility inside dark bars)
    tick = base.mark_tick(color='white', thickness=2, size=40).encode(
        y='median'
    )

    # Combine -> Facet -> Independent Scales
    chart = alt.layer(rule, bar, tick).facet(
        column=alt.Column('Feature', header=alt.Header(titleOrient="bottom", labelOrient="bottom"))
    ).resolve_scale(
        y='independent'
    ).properties(
        title=title
    )

    return chart


def plot_friends_vs_explicit(df):
    """
    Generates the first plot: Neutral Links vs Explicit Enemies.
    Works even if 'potential_mislabeled' (Hidden Hostility) is missing.
    """
    return _compute_and_plot(
        df,
        categories_to_include=['Neutral Links', 'Explicit Enemy'],
        title="Linguistic Fingerprint: Neutral Links vs Explicit Enemies"
    )


def plot_full_spectrum(df):
    """
    Generates the second plot: Neutral vs Hidden vs Explicit.
    Requires 'potential_mislabeled' to exist.
    """
    if 'potential_mislabeled' not in df.columns:
         raise ValueError("Cannot plot Hidden Hostility: 'potential_mislabeled' column is missing. Run your detection model first.")

    return _compute_and_plot(
        df,
        categories_to_include=['Neutral Links', 'Hidden Hostility', 'Explicit Enemy'],
        title="Linguistic Fingerprint: Neutral vs Hidden vs Explicit"
    )







def plot_precision_yield_curve(df, label_col="LINK_SENTIMENT", prob_col="p_true_hostile"):
    """
    Plots Precision vs. Yield to find the optimal threshold 'Elbow'.
    """
    # Color Palette for Orange/Gray Website
    COLOR_YIELD = "#264653"  # Deep Indigo
    COLOR_PREC  = "#2a9d8f"  # Vibrant Teal
    COLOR_GUIDE = "#8d99ae"  # Warm Slate Gray
    TEXT_COLOR  = "#2b2d42"  # Professional Dark Charcoal
    
    # Setup Data
    is_unlabeled = (df[label_col] != -1).to_numpy()
    probs = df[prob_col].to_numpy()
    
    # Define range of thresholds to test
    thresholds = np.linspace(0.50, 0.995, 100)
    
    precisions = []
    yields = []
    
    # Calculate Metrics for each Threshold
    for t in thresholds:
        flagged_mask = is_unlabeled & (probs >= t)
        n_flagged = flagged_mask.sum()
        
        if n_flagged > 0:
            avg_precision = probs[flagged_mask].mean()
        else:
            avg_precision = 1.0
            
        precisions.append(avg_precision)
        yields.append(n_flagged)
        
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Yield (Number of Flags) on Left Axis
    ax1.set_xlabel('Probability Threshold ($t$)', color=TEXT_COLOR, fontsize=11)
    ax1.set_ylabel('Number of Flagged Interactions (Yield)', color=COLOR_YIELD, fontweight='bold')
    ax1.plot(thresholds, yields, color=COLOR_YIELD, linewidth=3, label='Yield')
    ax1.tick_params(axis='y', labelcolor=COLOR_YIELD)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # Plot Precision on Right Axis
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Estimated Precision ($E[p]$)', color=COLOR_PREC, fontweight='bold')
    ax2.plot(thresholds, precisions, color=COLOR_PREC, linewidth=3, linestyle='--', label='Precision')
    ax2.tick_params(axis='y', labelcolor=COLOR_PREC)
    
    # Highlight Specific Candidates
    targets = [0.80, 0.85, 0.90, 0.95]
    prec_array = np.array(precisions)
    
    print(f"{'Target Prec':<15} | {'Threshold':<10} | {'Yield (Flags)':<15}")
    print("-" * 45)
    
    for target in targets:
        idx = np.where(prec_array >= target)[0]
        if len(idx) > 0:
            i = idx[0]
            t_star = thresholds[i]
            y_star = yields[i]
            
            # Draw vertical guide lines
            ax1.axvline(t_star, color=COLOR_GUIDE, linestyle=':', alpha=0.7)
            ax1.text(t_star, y_star, f" P~{target}", 
                     rotation=90, verticalalignment='bottom', 
                     color=TEXT_COLOR, fontsize=10, fontweight='bold')
            
            print(f"{target:<15} | {t_star:.3f}      | {y_star:<15}")
            
    plt.title("Elbow Method: Precision vs. Yield Trade-off", fontsize=14, pad=20, color=TEXT_COLOR, fontweight='bold')
    
    # Cleaning up spines
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.show()



def plot_variance_stabilization(df, min_n=1, max_n=50):
    """
    Plots the variance of pair-level sentiment against interaction count (N).
    Helps justify 'min_activity_per_pair'.
    """
    # High-contrast palette for Orange/Gray UI
    MAIN_LINE = "#1d3557"   # Deep Navy
    CUTOFF_COLOR = "#2a9d8f" # Vibrant Teal
    TEXT_COLOR = "#2b2d42"   # Dark Charcoal
    GRID_COLOR = "#e0e0e0"

    # Aggregate stats per Unordered Pair
    pair_stats = (
        df.groupby(["pair_a", "pair_b"])
          .agg(
              n_interactions=("LINK_SENTIMENT", "count"),
              neg_fraction=("LINK_SENTIMENT", lambda x: (x == -1).mean())
          )
    )
    
    pair_stats = pair_stats[pair_stats["n_interactions"].between(min_n, max_n)]

    variances = (
        pair_stats.groupby("n_interactions")["neg_fraction"]
                  .std()
                  .reset_index(name="sentiment_std")
    )
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Main Trend Line: Increased width and cleaner markers
    ax.plot(
        variances["n_interactions"], 
        variances["sentiment_std"], 
        marker='o', 
        markerfacecolor='white', # Hollow look makes it feel "lighter"
        markeredgewidth=2,
        markersize=8,
        linestyle='-', 
        linewidth=3, 
        color=MAIN_LINE,
        label="Volatility (Std Dev)"
    )
    
    # Highlight potential cutoffs
    for cutoff in [3, 5, 10]:
        if cutoff <= max_n:
            val = variances.loc[variances["n_interactions"] == cutoff, "sentiment_std"]
            if not val.empty:
                # Teal lines stand out beautifully against orange/gray backgrounds
                ax.axvline(cutoff, color=CUTOFF_COLOR, linestyle='--', alpha=0.7, linewidth=1.5)
                ax.text(
                    cutoff + 0.5, val.values[0], 
                    f" N={cutoff}", 
                    verticalalignment='bottom', 
                    color=CUTOFF_COLOR,
                    fontweight='bold',
                    fontsize=10
                )

    # Titles and Labels
    ax.set_title("Variance Stabilization: Volatility vs. Interaction Count", 
                 fontsize=14, pad=15, color=TEXT_COLOR, fontweight='bold')
    ax.set_xlabel("Number of Interactions per Pair (N)", color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Std. Dev. of Negative Fraction", color=TEXT_COLOR, labelpad=10)
    
    # Minimalist Grid and Spines
    ax.grid(True, axis='both', color=GRID_COLOR, linestyle='-', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    
    plt.tight_layout()
    plt.show()



def plot_source_bias(df, prob_col="p_true_hostile", threshold=0.85):
    """
    Plots the cumulative contribution of sources to the dataset to detect bias.
    Helps justify 'per_source_cap'.
    """
    # Custom Palette for Orange/Gray UI
    BAR_COLOR = "#4361ee"     # High-contrast Royal Blue
    CAP_LINE_COLOR = "#f72585" # Vivid Magenta (pops against orange/gray)
    TEXT_COLOR = "#2b2d42"     # Dark Charcoal
    
    # Identify Potential Flags
    potential_flags = df[df[prob_col] >= threshold].copy()
    total_flags = len(potential_flags)
    
    # Count Flags per Source
    source_counts = potential_flags["SOURCE_SUBREDDIT"].value_counts().reset_index()
    source_counts.columns = ["source", "count"]
    
    # Calculate percentage contribution
    source_counts["percent"] = source_counts["count"] / total_flags
    
    # Print Key Stats
    top_1 = source_counts.iloc[0]
    top_5_sum = source_counts.iloc[:5]["percent"].sum()
    
    print(f"Total Flagged Interactions: {total_flags}")
    print(f"Top 1 Contributor: r/{top_1['source']} ({top_1['percent']:.1%} of total)")
    print(f"Top 5 Contributors Combined: {top_5_sum:.1%} of total")
    
    # Plotting 
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot top 30 sources as a bar chart
    top_n = 30
    to_plot = source_counts.head(top_n)
    
    # Use a blue palette that contrasts with the orange website
    sns.barplot(
        x=to_plot.index, 
        y=to_plot["percent"], 
        color=BAR_COLOR, 
        ax=ax,
        alpha=0.85,
        edgecolor=TEXT_COLOR,
        linewidth=0.5
    )
    
    # Draw the "Cap" line - Magenta stands out best against orange background
    cap_val = 0.02
    ax.axhline(
        cap_val, 
        color=CAP_LINE_COLOR, 
        linestyle='--', 
        linewidth=2.5,
        label=f'Proposed Cap ({cap_val:.0%})'
    )
    
    # Titles and Labels
    ax.set_title(f"Source Bias: Top {top_n} Contributors to Hidden Hostility", 
                 fontsize=14, fontweight='bold', color=TEXT_COLOR, pad=20)
    ax.set_xlabel("Rank of Source Subreddit", color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Fraction of Total Flags", color=TEXT_COLOR, labelpad=10)
    
    # Styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis='y', linestyle=':', alpha=0.4, color=TEXT_COLOR)
    
    ax.legend(frameon=True, facecolor='white', edgecolor=TEXT_COLOR)
    
    plt.tight_layout()
    plt.show()



def plot_threshold_selection(df_scored, adaptive_flips, 
                                 p_certainty=0.99, 
                                 p_suspicion=0.85, 
                                 s_quantile=0.95):
    """
    Generates a scatter plot visualizing the 3-threshold decision logic 
    (Certainty, Suspicion, and Speed Corroboration).
    """
    
    alt.data_transformers.disable_max_rows()
    
    # Calculate the dynamic s_flip threshold based on the provided quantile
    s_threshold = adaptive_flips['s_flip'].quantile(s_quantile)
    
    # SAMPLE DATA 
    if len(df_scored) > 5000:
        plot_df = df_scored.sample(n=5000, random_state=42).copy()
    else:
        plot_df = df_scored.copy()

    # CLASSIFY REGIONS 
    def classify_region(row):
        p = row['p_true_hostile']
        s = row['s_flip']
        
        if p >= p_certainty:
            return "1. Certainty Zone (Accepted)"
        elif p >= p_suspicion and s >= s_threshold:
            return "2. Corroborated Zone (Accepted)"
        elif p >= p_suspicion and s < s_threshold:
            return "3. Rejected Zone (The Filter)"
        else:
            return "4. Safe Zone (Noise)"

    plot_df['Region'] = plot_df.apply(classify_region, axis=1)

    # BASE CHART 
    base = alt.Chart(plot_df).encode(
        x=alt.X('p_true_hostile', 
                scale=alt.Scale(domain=[0.5, 1.0]), 
                title='Linguistic Probability (p)'),
        y=alt.Y('s_flip', 
                scale=alt.Scale(domain=[0, 1.0]), 
                title='Reaction Speed (s_flip)'),
        tooltip=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 
                 alt.Tooltip('p_true_hostile', format='.3f'), 
                 alt.Tooltip('s_flip', format='.3f'), 
                 'Region']
    )

    # Scatter points
    points = base.mark_circle(size=60, opacity=0.5).encode(
        color=alt.Color('Region', 
                        scale=alt.Scale(domain=[
                            "1. Certainty Zone (Accepted)", 
                            "2. Corroborated Zone (Accepted)",
                            "3. Rejected Zone (The Filter)",
                            "4. Safe Zone (Noise)"
                        ],
                        range=['#1f77b4', '#ff7f0e', '#d62728', 'lightgrey'])),
        # Add stroke to highlight accepted points
        stroke=alt.condition(
            (alt.datum.Region == "1. Certainty Zone (Accepted)") | (alt.datum.Region == "2. Corroborated Zone (Accepted)"),
            alt.value('black'),
            alt.value(None)
        )
    )

    # DECISION BOUNDARIES 

    # A. Vertical Line: Certainty
    rule_certain = alt.Chart(pd.DataFrame({'x': [p_certainty]})).mark_rule(
        color='black', strokeDash=[5,5]
    ).encode(x='x')
    
    text_certain = alt.Chart(pd.DataFrame({
        'x': [p_certainty], 'y': [0.5], 'text': [f'Certainty (p={p_certainty})']
    })).mark_text(dx=5, angle=270).encode(x='x', y='y', text='text')

    # B. Vertical Line: Suspicion
    rule_suspicion = alt.Chart(pd.DataFrame({'x': [p_suspicion]})).mark_rule(
        color='black', strokeDash=[5,5]
    ).encode(x='x')
    
    text_suspicion = alt.Chart(pd.DataFrame({
        'x': [p_suspicion], 'y': [0.5], 'text': [f'Suspicion (p={p_suspicion})']
    })).mark_text(dx=-5, angle=270).encode(x='x', y='y', text='text')

    # C. Horizontal Segment: Speed Threshold
    # Only drawn between p_suspicion and p_certainty
    segment_data = pd.DataFrame({
        'x': [p_suspicion, p_certainty], 
        'y': [s_threshold, s_threshold]
    })
    segment_rule = alt.Chart(segment_data).mark_line(
        color='black', strokeDash=[5,5]
    ).encode(x='x', y='y')
    
    text_sflip = alt.Chart(pd.DataFrame({
        'x': [p_suspicion + (p_certainty - p_suspicion)/2],
        'y': [s_threshold], 
        'text': [f'Speed Cutoff (s={s_threshold:.2f})']
    })).mark_text(dy=-10, fontWeight='bold').encode(x='x', y='y', text='text')

    # COMPOSE 
    final_chart = (points + rule_certain + text_certain + rule_suspicion + text_suspicion + segment_rule + text_sflip).properties(
        title=f"Data-Driven Justification: The 3 Thresholds (Quantile {s_quantile*100:.0f}%)",
        width=600,
        height=500
    ).interactive()

    return final_chart


def plot_hostility_dashboard(df_input):
    """
    Takes the final hostility dataframe, processes it to identify Explicit vs Hidden hostility,
    filters for the top 50 aggressors, and returns an interactive Altair dashboard.
    """

    df = df_input.copy()

    # Dashboard Palette for Orange/Gray UI
    COLOR_EXPLICIT = "#264653"  # Deep Indigo/Navy
    COLOR_HIDDEN   = "#7209b7"  # Vivid Purple
    
    # Helper to map wide columns to the same colors
    RANGE_COLORS = [COLOR_EXPLICIT, COLOR_HIDDEN]

    def define_hostility_type(row):
        if row['LINK_SENTIMENT'] == -1:
            return 'Explicit Hostility'
        elif row['is_hidden_hostility'] == 1:
            return 'Hidden Hostility'
        else:
            return 'Non-Hostile'

    df['Hostility_Type'] = df.apply(define_hostility_type, axis=1)
    df_hostile = df[df['Hostility_Type'] != 'Non-Hostile'].copy()

    # Aggregating
    df_agg = df_hostile.groupby(['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'Hostility_Type']).size().reset_index(name='Count')

    # Pivot for specific ranking logic
    df_pivot = df_agg.pivot_table(index=['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT'], 
                                  columns='Hostility_Type', 
                                  values='Count', 
                                  fill_value=0).reset_index()
    
    if 'Explicit Hostility' not in df_pivot.columns: df_pivot['Explicit Hostility'] = 0
    if 'Hidden Hostility' not in df_pivot.columns: df_pivot['Hidden Hostility'] = 0

    df_pivot.rename(columns={'Explicit Hostility': 'Explicit_Count', 'Hidden Hostility': 'Hidden_Count'}, inplace=True)

    # Filter for Top 50 Aggressors
    source_totals = df_agg.groupby('SOURCE_SUBREDDIT')['Count'].sum().reset_index()
    top_sources_list = source_totals.sort_values('Count', ascending=False).head(50)['SOURCE_SUBREDDIT'].tolist()

    df_explorer_long = df_agg[df_agg['SOURCE_SUBREDDIT'].isin(top_sources_list)]
    df_explorer_wide = df_pivot[df_pivot['SOURCE_SUBREDDIT'].isin(top_sources_list)] 

    selection = alt.selection_point(fields=['SOURCE_SUBREDDIT'])

    # LEFT PANE: The Aggressors
    left_chart = alt.Chart(df_explorer_long).mark_bar().encode(
        y=alt.Y('SOURCE_SUBREDDIT', sort=top_sources_list, title='Top 50 Aggressors'),
        x=alt.X('sum(Count)', title='Total Hostile Links'),
        color=alt.Color('Hostility_Type', 
                        scale=alt.Scale(domain=['Explicit Hostility', 'Hidden Hostility'], 
                                        range=RANGE_COLORS), 
                        legend=alt.Legend(title="Type", orient='top')),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.25)),
        tooltip=['SOURCE_SUBREDDIT', 'Hostility_Type', 'sum(Count)']
    ).add_params(
        selection
    ).properties(
        title='1. Select an Aggressor',
        width=250,
        height=600
    )

    # RIGHT PANE 1: Ranked by EXPLICIT Hostility 
    explicit_rank_chart = alt.Chart(df_explorer_wide).transform_filter(
        selection
    ).transform_window(
        rank='rank(Explicit_Count)',
        sort=[alt.SortField('Explicit_Count', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 10
    ).transform_fold(
        ['Explicit_Count', 'Hidden_Count'],
        as_=['Hostility_Type', 'Count']
    ).mark_bar().encode(
        y=alt.Y('TARGET_SUBREDDIT', sort=alt.EncodingSortField(field="Explicit_Count", order="descending"), title=None),
        x=alt.X('Count:Q', title='Interactions (Sorted by Explicit)'),
        color=alt.Color('Hostility_Type:N', scale=alt.Scale(domain=['Explicit_Count', 'Hidden_Count'], range=RANGE_COLORS)),
        tooltip=['TARGET_SUBREDDIT', 'Hostility_Type:N', 'Count:Q']
    ).properties(
        title='2. Top Targets of EXPLICIT Hostility',
        width=300,
        height=250
    )

    # RIGHT PANE 2: Ranked by HIDDEN Hostility 
    hidden_rank_chart = alt.Chart(df_explorer_wide).transform_filter(
        selection
    ).transform_window(
        rank='rank(Hidden_Count)',
        sort=[alt.SortField('Hidden_Count', order='descending')]
    ).transform_filter(
        alt.datum.rank <= 10
    ).transform_fold(
        ['Explicit_Count', 'Hidden_Count'],
        as_=['Hostility_Type', 'Count']
    ).mark_bar().encode(
        y=alt.Y('TARGET_SUBREDDIT', sort=alt.EncodingSortField(field="Hidden_Count", order="descending"), title=None),
        x=alt.X('Count:Q', title='Interactions (Sorted by Hidden)'),
        color=alt.Color('Hostility_Type:N', scale=alt.Scale(domain=['Explicit_Count', 'Hidden_Count'], range=RANGE_COLORS)),
        tooltip=['TARGET_SUBREDDIT', 'Hostility_Type:N', 'Count:Q']
    ).properties(
        title='3. Top Targets of HIDDEN Hostility',
        width=300,
        height=250
    )

    # Styling and Combine
    dashboard = (left_chart | (explicit_rank_chart & hidden_rank_chart)).configure_title(
        fontSize=16,
        fontWeight='bold',
        anchor='middle'
    ).configure_axis(
        labelFontSize=11,
        titleFontSize=12
    )

    return dashboard
