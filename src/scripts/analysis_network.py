import networkx as nx
import pandas as pd
from tqdm import tqdm


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    Builds a directed signed graph from a DataFrame containing:
      - SOURCE_SUBREDDIT
      - TARGET_SUBREDDIT
      - LINK_SENTIMENT (as edge weight)
    """
    G = nx.DiGraph()
    G.add_weighted_edges_from(df[["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "LINK_SENTIMENT"]].values)

    print(f"Directed signed graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G


def compute_node_metrics(G: nx.DiGraph) -> pd.DataFrame:
    """
    Computes main node-level metrics for the graph:
    - degree, in-degree, out-degree
    - degree centrality
    - betweenness centrality
    - PageRank centrality
    """

    print("Computing node-level metrics...")

    degree = dict(G.degree())
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    degree_centrality = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G, k=min(1000, len(G)), seed=42)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100)

    metrics_df = pd.DataFrame({
        "subreddit": list(G.nodes()),
        "degree": [degree[n] for n in G.nodes()],
        "in_degree": [in_degree[n] for n in G.nodes()],
        "out_degree": [out_degree[n] for n in G.nodes()],
        "degree_centrality": [degree_centrality[n] for n in G.nodes()],
        "betweenness": [betweenness[n] for n in G.nodes()],
        "pagerank": [pagerank[n] for n in G.nodes()]
    })

    metrics_df.to_csv("reddit_node_metrics.csv", index=False)
    print("Node metrics saved to reddit_node_metrics.csv")

    return metrics_df


def compute_global_balance_index(G: nx.DiGraph) -> float | None:
    """
    Computes the proportion of balanced directed triads:
    A triad A→B→C with A→C is considered balanced if the product of edge signs = +1.

    Returns:
        - balance_index (float between 0 and 1)
        - None if no closed triads are found
    """
    triads = 0
    balanced = 0

    print("🔄 Computing Global Balance Index...")

    for a in tqdm(G.nodes, desc="Checking triads"):
        for b in G.successors(a):
            for c in G.successors(b):
                if G.has_edge(a, c):
                    triads += 1
                    s_ab = G[a][b]["weight"]
                    s_bc = G[b][c]["weight"]
                    s_ac = G[a][c]["weight"]
                    if s_ab * s_bc * s_ac == 1:
                        balanced += 1

    if triads == 0:
        print("No triads found for balance computation.")
        return None

    balance_index = balanced / triads
    print(f"Global Balance Index: {balance_index:.3f}")
    return balance_index


def analyze_network(df: pd.DataFrame) -> dict:
    """
    Main function: builds the graph, computes node metrics and the Global Balance Index.
    Returns a dictionary ready to be integrated into `results`.
    """

    # 1. Build the graph
    G = build_graph(df)

    # 2. Compute node-level metrics
    metrics_df = compute_node_metrics(G)

    # 3. Compute the Global Balance Index
    balance_index = compute_global_balance_index(G)

    # 4. Compute the average global degree
    average_degree = G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    print(f"Average node degree: {average_degree:.2f}")

    # 5. Return structured results
    return {
        "graph": G,
        "metrics": metrics_df,
        "balance_index": balance_index,
        "average_degree": average_degree
    }


import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd

def plot_network_summary(results: dict, original_df: pd.DataFrame | None = None):
    """
    Generates a visual summary of the inter-subreddit network analysis.

    Args:
        results (dict): Output dictionary from `analyze_network()`, containing:
            - results["network_analysis"]["metrics"]: node-level metrics DataFrame
            - results["network_analysis"]["balance_index"]: global balance index
            - results["network_analysis"]["graph"]: NetworkX DiGraph
        original_df (pd.DataFrame, optional): Original dataframe with LINK_SENTIMENT
                                              for sentiment distribution plot.
    """

    metrics_df = results["network_analysis"]["metrics"]
    balance_index = results["network_analysis"]["balance_index"]

    # --- Degree distribution ---
    plt.figure(figsize=(6, 4))
    sns.histplot(metrics_df["degree"], bins=50, log_scale=True, color="#4682B4")
    plt.title("Degree Distribution (Log Scale)")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

    # --- Top 10 subreddits by degree ---
    top_degree = metrics_df.sort_values("degree", ascending=False).head(10)
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top_degree,
        x="degree",
        y="subreddit",
        hue="subreddit",          # fixes palette warning
        palette="crest",
        legend=False
    )
    plt.title("Top 10 Subreddits by Total Degree")
    plt.xlabel("Degree")
    plt.ylabel("Subreddit")
    plt.tight_layout()
    plt.show()

    # --- Degree centrality vs PageRank ---
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=metrics_df,
        x="degree_centrality",
        y="pagerank",
        alpha=0.6,
        color="#1D7874"
    )
    plt.title("Degree Centrality vs PageRank")
    plt.xlabel("Degree Centrality")
    plt.ylabel("PageRank")
    plt.tight_layout()
    plt.show()

    # --- Global balance index ---
    if balance_index is not None:
        plt.figure(figsize=(5, 4))
        plt.bar(["Balance Index"], [balance_index], color="#2A9D8F")
        plt.ylim(0, 1)
        plt.title(f"Global Balance Index = {balance_index:.3f}")
        plt.ylabel("Proportion of Balanced Triads")
        plt.tight_layout()
        plt.show()


