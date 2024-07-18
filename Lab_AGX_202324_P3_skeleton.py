import networkx as nx
from collections import Counter
import community as community_louvain

# Marino Oliveros Blanco NIU:1668563
# Pere Mayol Carbonell NIU:1669503


# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #

# Function to get artist names from IDs using the graph node attributes
def get_artist_names(graph, artist_ids):
    return [(artist_id, graph.nodes[artist_id].get('name', 'Unknown Artist')) for artist_id in artist_ids]

# Function to find the minimum cost
def min_ad_cost(g):
    # Find all strongly connected components in the graph
    sccs = list(nx.strongly_connected_components(g))
    min_cost = len(sccs)  # Each SCC (strongly connected comp) needs at least one ad to ensure coverage
    return min_cost

# Function to select the most central artists within a budget to maximize spread (explained why in the report)
def select_artists_for_budget(g, budget):
    sccs = list(nx.strongly_connected_components(g))
    scc_central_nodes = []
    
    for scc in sccs:
        subgraph = g.subgraph(scc)
        centrality = nx.degree_centrality(subgraph)
        most_central_node = max(centrality, key=centrality.get)
        scc_central_nodes.append((most_central_node, centrality[most_central_node]))
    
    # Sort the central nodes by their centrality in descending order
    scc_central_nodes = sorted(scc_central_nodes, key=lambda x: x[1], reverse=True)
    
    selected_artists = []
    total_cost = 0
    artists_per_cost = 100  # Each artist costs 100 euros
    max_artists = budget // artists_per_cost  # Calculate maximum number of artists we can afford
    
    for artist, _ in scc_central_nodes:
        if total_cost + artists_per_cost <= budget:
            selected_artists.append(artist)
            total_cost += artists_per_cost
        if len(selected_artists) >= max_artists:
            break
    
    return selected_artists

# --------------- END OF AUXILIARY FUNCTIONS ------------------ #


def num_common_nodes(*args):
    """
    Return the number of common nodes between a set of graphs.

    :param args: (an undetermined number of) networkx graphs.
    :return: an integer, number of common nodes.
    """
    common_nodes = set(args[0].nodes())
    for graph in args[1:]:
        common_nodes &= set(graph.nodes())
    return len(common_nodes)

def get_degree_distribution(g: nx.Graph) -> dict:
    """
    Get the degree distribution of the graph.

    :param g: networkx graph.
    :return: dictionary with degree distribution (keys are degrees, values are number of occurrences).
    """
    degree_count = Counter(dict(g.degree()).values())
    return dict(degree_count)

def get_k_most_central(g: nx.Graph, metric: str, num_nodes: int) -> list:
    """
    Get the k most central nodes in the graph.

    :param g: networkx graph.
    :param metric: centrality metric. Can be (at least) 'degree', 'betweenness', 'closeness' or 'eigenvector'.
    :param num_nodes: number of nodes to return.
    :return: list with the top num_nodes nodes.
    """
    centrality_metrics = {
        'degree': nx.degree_centrality,
        'betweenness': nx.betweenness_centrality,
        'closeness': nx.closeness_centrality,
        'eigenvector': nx.eigenvector_centrality
    }

    if metric not in centrality_metrics:
        raise ValueError(f"Metric {metric} is not supported.")
    
    centrality = centrality_metrics[metric](g)
    sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    return [node for node, _ in sorted_nodes[:num_nodes]]

def find_cliques(g: nx.Graph, min_size_clique: int) -> tuple:
    """
    Find cliques in the graph g with size at least min_size_clique.

    :param g: networkx graph.
    :param min_size_clique: minimum size of the cliques to find.
    :return: two-element tuple, list of cliques (each clique is a list of nodes) and list of nodes in any of the cliques.
    """
    cliques = [clique for clique in nx.find_cliques(g) if len(clique) >= min_size_clique]
    nodes_in_cliques = set(node for clique in cliques for node in clique)
    return cliques, list(nodes_in_cliques)

def detect_communities(g: nx.Graph, method: str) -> tuple:
    """
    Detect communities in the graph g using the specified method.

    :param g: a networkx graph.
    :param method: string with the name of the method to use. Can be (at least) 'girvan-newman' or 'louvain'.
    :return: two-element tuple, list of communities (each community is a list of nodes) and modularity of the partition.
    """
    if method == 'girvan-newman': # use Girvan-Newman method as done in class
        communities_generator = nx.community.girvan_newman(g)
        top_level_communities = next(communities_generator)
        communities = [list(c) for c in top_level_communities]
    elif method == 'louvain': # Lovain/Leuven (little town near Brussels) method an algorithm to detect communities in large networks (for it you have to 'pip install python-louvain')

        if g.is_directed():
            g = g.to_undirected()
        partition = community_louvain.best_partition(g)
        communities_dict = {}
        for node, comm in partition.items():
            communities_dict.setdefault(comm, []).append(node)
        communities = list(communities_dict.values())
    else:
        raise ValueError(f"Method {method} is not supported.")

    modularity = nx.community.modularity(g, communities)
    return communities, modularity


# ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #

def main():
    gD = nx.read_graphml('gD.graphml')
    gB = nx.read_graphml('gB.graphml')
    gBp = nx.read_graphml('gBp.graphml')
    gw = nx.read_graphml('gw.graphml')
    hB = nx.read_graphml('hB.graphml')
    gDp = nx.read_graphml('gDp.graphml')


    # Number of common nodes between gB and gD
    common_nodes_gB_gD = num_common_nodes(gB, gD)
    print(f"Number of common nodes between gB and gD: {common_nodes_gB_gD}")

    # Number of common nodes between gB and gBp
    common_nodes_gB_gBp = num_common_nodes(gB, gBp)
    print(f"Number of common nodes between gB and gBp: {common_nodes_gB_gBp}")

    # 25 most central nodes in gBp using degree centrality and betweenness centrality
    degree_central_nodes = get_k_most_central(gBp, 'degree', 25)
    betweenness_central_nodes = get_k_most_central(gBp, 'betweenness', 25)

    common_central_nodes = set(degree_central_nodes) & set(betweenness_central_nodes)
    print(f"Number of common nodes between degree and betweenness centrality: {len(common_central_nodes)}")

    # Find cliques of size greater than or equal to min_size_clique in gBp and gDp
    min_size_clique = 7  # Max num with at least 2 cliques (did this by trial an error)
    cliques_gBp, nodes_in_cliques_gBp = find_cliques(gBp, min_size_clique)
    cliques_gDp, nodes_in_cliques_gDp = find_cliques(gDp, min_size_clique)

    print(f"Number of cliques in gBp: {len(cliques_gBp)}, Total nodes in cliques: {len(nodes_in_cliques_gBp)}")
    print(f"Number of cliques in gDp: {len(cliques_gDp)}, Total nodes in cliques: {len(nodes_in_cliques_gDp)}")

    # Analyze one of the largest cliques
    largest_clique_gBp = max(cliques_gBp, key=len)
    largest_clique_names = get_artist_names(gBp, largest_clique_gBp)
    print(f"Largest clique in gBp: {largest_clique_names}")

    # Detect communities in gD
    try:
        communities_gD, modularity_gD = detect_communities(gD, 'louvain')
        communities_gD_with_names = [get_artist_names(gD, community) for community in communities_gD]
        print(f"Communities in gD: {communities_gD_with_names}, Modularity: {modularity_gD}")
    except ModuleNotFoundError as e:
        print(e)


    # Minimum cost for gB and gD
    min_cost_gB = min_ad_cost(gB)
    min_cost_gD = min_ad_cost(gD)
    print(f"Minimum cost for gB: {min_cost_gB * 100} euros")
    print(f"Minimum cost for gD: {min_cost_gD * 100} euros")

    # Selected artists within a 400 euros budget
    selected_artists_gB = select_artists_for_budget(gB, 400)
    selected_artists_gD = select_artists_for_budget(gD, 400)

    selected_artists_names_gB = get_artist_names(gB, selected_artists_gB)
    selected_artists_names_gD = get_artist_names(gD, selected_artists_gD)

    print(f"Selected artists for gB within 400 euros budget: {selected_artists_names_gB}")
    print(f"Selected artists for gD within 400 euros budget: {selected_artists_names_gD}")

    # Calculate minimum hops between two artists in gB
    def min_hops_between_artists(g, start_artist, target_artist):
        if start_artist not in g:
            raise nx.NodeNotFound(f"Source {start_artist} is not in the graph")
        if target_artist not in g:
            raise nx.NodeNotFound(f"Target {target_artist} is not in the graph")
        
        if nx.has_path(g, start_artist, target_artist):
            path = nx.shortest_path(g, source=start_artist, target=target_artist)
            return len(path) - 1, path
        else:
            return float('inf'), []

    # Minimum hops from "Taylor Swift" to "THE DRIVER ERA"
    start_artist = '06HL4z0CvFAxyc27GXpf02'  # Taylor Swift id
    target_artist = '5bmqhxWk9SEFDGIzWpSjVJ'  # THE DRIVER ERA id
    try:
        min_hops, path = min_hops_between_artists(gB, start_artist, target_artist)
        path_with_names = get_artist_names(gB, path)
        print(f"Minimum hops from Taylor Swift to THE DRIVER ERA: {min_hops}")
        print(f"Path with artist names: {path_with_names}")
    except nx.NodeNotFound as e:
        print(e)

if __name__ == '__main__':
    main()