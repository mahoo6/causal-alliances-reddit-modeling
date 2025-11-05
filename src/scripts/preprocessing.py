import pandas as pd

# === List of the 86 properties ===
property_names = [
    "n_chars", "n_chars_no_space", "frac_alpha", "frac_digits", "frac_upper",
    "frac_spaces", "frac_special", "n_words", "n_unique_words", "n_long_words",
    "avg_word_len", "n_unique_stopwords", "frac_stopwords", "n_sentences",
    "n_long_sentences", "avg_chars_per_sentence", "avg_words_per_sentence",
    "readability_index", "VADER_pos", "VADER_neg", "VADER_compound",
    "LIWC_Funct", "LIWC_Pronoun", "LIWC_Ppron", "LIWC_I", "LIWC_We",
    "LIWC_You", "LIWC_SheHe", "LIWC_They", "LIWC_Ipron", "LIWC_Article",
    "LIWC_Verbs", "LIWC_AuxVb", "LIWC_Past", "LIWC_Present", "LIWC_Future",
    "LIWC_Adverbs", "LIWC_Prep", "LIWC_Conj", "LIWC_Negate", "LIWC_Quant",
    "LIWC_Numbers", "LIWC_Swear", "LIWC_Social", "LIWC_Family", "LIWC_Friends",
    "LIWC_Humans", "LIWC_Affect", "LIWC_Posemo", "LIWC_Negemo", "LIWC_Anx",
    "LIWC_Anger", "LIWC_Sad", "LIWC_CogMech", "LIWC_Insight", "LIWC_Cause",
    "LIWC_Discrep", "LIWC_Tentat", "LIWC_Certain", "LIWC_Inhib", "LIWC_Incl",
    "LIWC_Excl", "LIWC_Percept", "LIWC_See", "LIWC_Hear", "LIWC_Feel",
    "LIWC_Bio", "LIWC_Body", "LIWC_Health", "LIWC_Sexual", "LIWC_Ingest",
    "LIWC_Relativ", "LIWC_Motion", "LIWC_Space", "LIWC_Time", "LIWC_Work",
    "LIWC_Achiev", "LIWC_Leisure", "LIWC_Home", "LIWC_Money", "LIWC_Relig",
    "LIWC_Death", "LIWC_Assent", "LIWC_Dissent", "LIWC_Nonflu", "LIWC_Filler"
]


def load_data(title_path: str, body_path: str, embeddings_path: str = None):
    """Load the Reddit Hyperlinks datasets (title, body) and optionally subreddit embeddings."""
    title_df = pd.read_csv(title_path, sep='\t')
    body_df = pd.read_csv(body_path, sep='\t')
    print(f"Loaded title ({title_df.shape}) and body ({body_df.shape}) datasets.")
    
    if embeddings_path:
        embeddings_df = pd.read_csv(embeddings_path, header=None)
        print(f"Loaded embeddings ({embeddings_df.shape}) dataset.")
        n_cols = embeddings_df.shape[1]
        embeddings_df.columns = ["subreddit"] + [f"emb{i}" for i in range(n_cols - 1)]
        for c in embeddings_df.columns[1:]:
            embeddings_df[c] = pd.to_numeric(embeddings_df[c], errors="coerce")
        return title_df, body_df, embeddings_df
    
    return title_df, body_df



def expand_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Split the PROPERTIES column into separate numeric columns."""
    props = df["PROPERTIES"].str.split(",", expand=True)
    props.columns = property_names[:props.shape[1]]
    props = props.apply(pd.to_numeric, errors="coerce")
    return pd.concat([df.drop(columns=["PROPERTIES"]), props], axis=1)

def clean_dataframe(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """
    Clean the data:
    - For hyperlink datasets: remove self-links, duplicates, and invalid dates.
    - For embedding datasets: drop empty subreddit names and duplicate rows.
    Prints the number of rows removed at each step.
    """
    initial_rows = len(df)
    print(f"\nCleaning {name or 'dataset'} ({initial_rows} rows)...")

    if {"SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP"}.issubset(df.columns):
        # --- Case 1: hyperlink datasets ---
        # Remove self-links
        before = len(df)
        df = df[df["SOURCE_SUBREDDIT"] != df["TARGET_SUBREDDIT"]]
        print(f"Removed self-links: {before - len(df)} rows")

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "TIMESTAMP"])
        print(f"Removed duplicates: {before - len(df)} rows")

        # Convert TIMESTAMP to datetime and drop invalid
        before = len(df)
        df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce")
        df = df.dropna(subset=["TIMESTAMP"])
        print(f"Removed invalid dates: {before - len(df)} rows")

    elif "subreddit" in df.columns:
        # --- Case 2: embeddings dataset ---
        # Remove empty subreddit names
        before = len(df)
        df = df.dropna(subset=["subreddit"])
        df = df[df["subreddit"].astype(str).str.strip() != ""]
        print(f"Removed empty subreddit names: {before - len(df)} rows")

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates(subset=["subreddit"])
        print(f"Removed duplicate subreddits: {before - len(df)} rows")

        # Optionally check embeddings for NaN values
        before = len(df)
        embedding_cols = [c for c in df.columns if c.startswith("emb")]
        nan_rows = df[embedding_cols].isna().any(axis=1).sum()
        if nan_rows > 0:
            df = df.dropna(subset=embedding_cols)
            print(f"Removed rows with NaN embeddings: {nan_rows} rows")
        else:
            print("No NaN embeddings found.")

    else:
        print("Warning: No recognizable subreddit columns to clean.")

    print(f"→ Final dataset size: {len(df)} rows (removed {initial_rows - len(df)} total)\n")
    return df



def combine_datasets(title_df: pd.DataFrame, body_df: pd.DataFrame) -> pd.DataFrame:
    """Combine the title/body datasets and add a link_source_type column."""
    title_df["link_source_type"] = "title"
    body_df["link_source_type"] = "body"
    combined = pd.concat([title_df, body_df], ignore_index=True)
    return combined

def visualize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Visualize the dataset"""    
    df["month"] = df["TIMESTAMP"].dt.to_period("M").astype(str)
    return df[["SOURCE_SUBREDDIT","TARGET_SUBREDDIT","POST_ID","TIMESTAMP","month","LINK_SENTIMENT"]].reset_index(drop=True)


