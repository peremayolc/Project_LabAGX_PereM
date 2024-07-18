import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt

# Marino Oliveros Blanco NIU:1668563
# Pere Mayol Carbonell NIU:1669503

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.
    """
    
    undirected_g = nx.Graph()  # Create an empty undirected graph
    
    # Add only bidirectional edges
    for u, v in g.edges():
        if g.has_edge(v, u):  # Check if there is a reverse edge
            undirected_g.add_edge(u, v)  # Add the edge to the undirected graph
    
    # Preserve node attributes
    for node in undirected_g.nodes():
        if 'name' not in g.nodes[node]:
            undirected_g.nodes[node]['name'] = node  # Assign the node itself as name if 'name' attribute is missing
        else:
            undirected_g.nodes[node]['name'] = g.nodes[node]['name']  # Preserve the 'name' attribute
    
    # Save the undirected graph to a file
    nx.write_graphml(undirected_g, out_filename)
    
    return undirected_g

def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree.

    :param g: a networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    # Keep nodes with degree >= min_degree
    nodes_to_keep = [node for node, degree in dict(g.degree()).items() if degree >= min_degree]
    pruned_g = g.subgraph(nodes_to_keep).copy()  # Create a subgraph with the selected nodes

    # Save the pruned graph to a file
    nx.write_graphml(pruned_g, out_filename)
    return pruned_g

def prune_low_weight_edges(g: nx.Graph, min_weight: float = None, min_percentile: int = None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    
    # Ensure only one of min_weight or min_percentile is specified
    if (min_weight is None and min_percentile is None) or (min_weight is not None and min_percentile is not None):
        raise ValueError("Specify either min_weight or min_percentile, but not both.")
    
    if min_percentile is not None:
        weights = [data['weight'] for u, v, data in g.edges(data=True)]  # Get all edge weights
        threshold = pd.Series(weights).quantile(min_percentile / 100)  # Compute the weight threshold based on percentile
    else:
        threshold = min_weight  # Use the specified min_weight as threshold
    
    # Keep edges with weight >= threshold
    edges_to_keep = [(u, v) for u, v, data in g.edges(data=True) if data['weight'] >= threshold]
    pruned_g = g.edge_subgraph(edges_to_keep).copy()  # Create a subgraph with the selected edges
    
    # Remove nodes with zero degree
    zero_degree_nodes = [node for node, degree in dict(pruned_g.degree()).items() if degree == 0]
    pruned_g.remove_nodes_from(zero_degree_nodes)
    
    # Preserve node attributes
    for node in pruned_g.nodes():
        if 'name' not in g.nodes[node]:
            pruned_g.nodes[node]['name'] = node  # Assign the node itself as name if 'name' attribute is missing
        else:
            pruned_g.nodes[node]['name'] = g.nodes[node]['name']  # Preserve the 'name' attribute
    
    # Save the pruned graph to a file if specified
    if out_filename:
        nx.write_graphml(pruned_g, out_filename)
    return pruned_g

def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # List of audio feature columns
    audio_feature_columns = ['danceability', 'energy', 'loudness', 'speechiness',
                             'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    # Compute the mean of audio features grouped by artist
    mean_features_df = tracks_df.groupby(['artist_id', 'artist_name'])[audio_feature_columns].mean().reset_index()
    return mean_features_df

def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.
    """
    feature_columns = artist_audio_features_df.select_dtypes(include='number').columns  # Select numerical feature columns
    
    # Compute the similarity matrix based on the specified metric
    if similarity == "cosine":
        similarity_matrix = cosine_similarity(artist_audio_features_df[feature_columns])
    elif similarity == "euclidean":
        similarity_matrix = euclidean_distances(artist_audio_features_df[feature_columns])
        similarity_matrix = 1 / (1 + similarity_matrix)  # Convert distances to similarity
    else:
        raise ValueError("Unknown similarity metric. Use 'cosine' or 'euclidean'.")
    
    similarity_graph = nx.Graph()  # Create an empty graph
    num_artists = artist_audio_features_df.shape[0]
    
    # Add edges with similarity weights
    for i in range(num_artists):
        for j in range(i + 1, num_artists):
            if i != j:  # Avoid self-loops (so artist's similarity is not compared to themeselves)
                similarity_graph.add_edge(artist_audio_features_df.iloc[i, 0], artist_audio_features_df.iloc[j, 0],
                                          weight=similarity_matrix[i, j])
                similarity_graph.nodes[artist_audio_features_df.iloc[i, 0]]['name'] = artist_audio_features_df.iloc[i, 1]
                similarity_graph.nodes[artist_audio_features_df.iloc[j, 0]]['name'] = artist_audio_features_df.iloc[j, 1]
    
    # Save the similarity graph to a file if specified
    if out_filename:
        nx.write_graphml(similarity_graph, out_filename)
    
    return similarity_graph

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #
def visualize_graph(graph, title):
    pos = nx.spring_layout(graph)  # Compute positions for nodes
    plt.figure(figsize=(10, 10))
    labels = nx.get_node_attributes(graph, 'name')  # Get node labels
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=50, font_size=8)  # Draw the graph
    plt.title(title)
    plt.show()

def print_graph_metrics(graph, name):
    # Print basic metrics of the graph
    print(f"Metrics for {name}:")
    print(f"Order (number of nodes): {graph.number_of_nodes()}")
    print(f"Size (number of edges): {graph.number_of_edges()}")
    print(f"Density: {nx.density(graph):.4f}")
    if nx.is_directed(graph):
        print(f"Strongly connected components: {nx.number_strongly_connected_components(graph)}")
        print(f"Weakly connected components: {nx.number_weakly_connected_components(graph)}")
    else:
        print(f"Connected components: {nx.number_connected_components(graph)}")

def find_most_least_similar_artists(graph):
    # Sort edges based on weight
    edges = sorted((u, v, d) for u, v, d in graph.edges(data=True) if u != v)  # Exclude self-loops
    most_similar = edges[0]  # Edge with the highest weight
    least_similar = edges[-1]  # Edge with the lowest weight
    print(f"Debug: Most similar edge weight: {most_similar[2]['weight']}")
    print(f"Debug: Least similar edge weight: {least_similar[2]['weight']}")
    return most_similar, least_similar

def find_most_least_similar_to_all(graph):
    # Compute average similarity for each node
    avg_weights = {node: sum(data['weight'] for _, _, data in graph.edges(node, data=True)) / graph.degree(node) 
                   for node in graph.nodes()}
    most_similar = max(avg_weights, key=avg_weights.get)  # Node with the highest average similarity
    least_similar = min(avg_weights, key=avg_weights.get)  # Node with the lowest average similarity
    return most_similar, least_similar

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


# --------------- MAIN FUNCTION TO RUN THE CODE --------------- #
def main():
    # Load the graphs
    gB = nx.read_graphml('gB.graphml')
    gD = nx.read_graphml('gD.graphml')

    # Load tracks data 
    tracks_df = pd.read_csv('songs.csv')
    tracks_df = pd.DataFrame(tracks_df)

    # Task 6(a): Generate undirected graphs g'B and g'D
    gB_undirected = retrieve_bidirectional_edges(gB, 'gBp.graphml')
    gD_undirected = retrieve_bidirectional_edges(gD, 'gDp.graphml')

    # Task 6(b): Compute mean audio features
    mean_audio_features_df = compute_mean_audio_features(tracks_df)
    print(mean_audio_features_df)

    # Create similarity graph
    similarity_graph = create_similarity_graph(mean_audio_features_df, 'cosine', 'gw.graphml')
    print("Graphs have been processed and saved successfully.")

  # Visualize the graphs
    visualize_graph(gB_undirected, "Undirected Graph g'B")
    visualize_graph(gD_undirected, "Undirected Graph g'D")
    visualize_graph(similarity_graph, "Similarity Graph")

    # Print metrics for each graph
    print_graph_metrics(gB_undirected, "Undirected Graph g'B")
    print_graph_metrics(gD_undirected, "Undirected Graph g'D")
    print_graph_metrics(similarity_graph, "Similarity Graph")

    # Find and print the most and least similar artists
    most_similar, least_similar = find_most_least_similar_artists(similarity_graph)
    print(f"Most similar artists: {similarity_graph.nodes[most_similar[0]]['name']} and {similarity_graph.nodes[most_similar[1]]['name']} with weight {most_similar[2]['weight']}")
    print(f"Least similar artists: {similarity_graph.nodes[least_similar[0]]['name']} and {similarity_graph.nodes[least_similar[1]]['name']} with weight {least_similar[2]['weight']}")

    # Find and print the artist most and least similar to all others
    most_similar_to_all, least_similar_to_all = find_most_least_similar_to_all(similarity_graph)
    print(f"Artist most similar to all others: {similarity_graph.nodes[most_similar_to_all]['name']}")
    print(f"Artist least similar to all others: {similarity_graph.nodes[least_similar_to_all]['name']}")

if __name__ == "__main__":
    main()