def print_df_summary(df, name="DataFrame"):
    """Displays a quick summary of the DataFrame (types and statistics)."""
    print(f"=== {name} Summary ===")
    print(df.info())
    print(df.describe())
