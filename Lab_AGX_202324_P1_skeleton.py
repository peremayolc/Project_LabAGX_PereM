import networkx as nx
import pandas as pd
import spotipy
from spotipy . oauth2 import SpotifyClientCredentials
import time


# Marino Oliveros Blanco NIU:1668563
# Pere Mayol Carbonell NIU:1669503


# ------- AUXILIARY FUNCTIONS ------- #
def degree_statistics(graph):
    in_degrees = [d for n, d in graph.in_degree()]
    out_degrees = [d for n, d in graph.out_degree()]
    
    stats = {
        "in_degree": {
            "min": min(in_degrees),
            "max": max(in_degrees),
            "median": sorted(in_degrees)[len(in_degrees)//2]
        },
        "out_degree": {
            "min": min(out_degrees),
            "max": max(out_degrees),
            "median": sorted(out_degrees)[len(out_degrees)//2]
        },
        "number_of_nodes": graph.number_of_nodes(),
        "number_of_edges": graph.number_of_edges()
    }
    return stats
# --------------- END OF AUXILIARY FUNCTIONS ------------------ #



def search_artist(sp: spotipy.client.Spotify, artist_name: str) -> str:
    """
    Search for an artist in Spotify.

    :param sp: spotipy client object
    :param artist_name: name to search for.
    :return: spotify artist id.
    """

    id = sp.search(artist_name, type='artist', limit=5)['artists']['items'][0]['id'] # Get the first result

    return id


def crawler(sp: spotipy.client.Spotify, seed: str, max_nodes_to_crawl: int, strategy: str = "BFS", 
            out_filename: str = "g.graphml") -> nx.DiGraph:
    """
    Crawl the Spotify artist graph, following related artists.

    :param sp: spotipy client object
    :param seed: starting artist id.
    :param max_nodes_to_crawl: maximum number of nodes to crawl.
    :param strategy: BFS or DFS.
    :param out_filename: name of the graphml output file.
    :return: networkx directed graph.
    """

    G = nx.DiGraph()
    visited = set()
    all_related_artists = set()
    api_call_count = 0  # Initialize the API call counter

    if strategy == "BFS":
        queue = [(seed, None)]  # BFS uses a queue
    else:
        stack = [(seed, None)]  # DFS uses a stack

    while len(visited) < max_nodes_to_crawl:
        if strategy == "BFS" and queue:
            current_artist_id, parent_id = queue.pop(0)
        elif strategy == "DFS" and stack:
            current_artist_id, parent_id = stack.pop()
        else:
            break

        if current_artist_id not in visited:
            visited.add(current_artist_id)
            try:
                artist_info = sp.artist(current_artist_id)
                api_call_count += 1  # Increment the API call counter
                print(f"API call {api_call_count}: Retrieved artist {current_artist_id}")
                G.add_node(current_artist_id, 
                           name=artist_info['name'],
                           followers=artist_info['followers']['total'],
                           popularity=artist_info['popularity'],
                           genres=";".join(artist_info['genres']))  # Convert list to string
                if parent_id:
                    G.add_edge(parent_id, current_artist_id)

                related_artists = sp.artist_related_artists(current_artist_id)['artists']
                api_call_count += 1  # Increment the API call counter
                print(f"API call {api_call_count}: Retrieved related artists for {current_artist_id}")
                for related_artist in related_artists:
                    if related_artist['id'] not in visited:
                        G.add_node(related_artist['id'],
                                   name=related_artist['name'],
                                   followers=related_artist['followers']['total'],
                                   popularity=related_artist['popularity'],
                                   genres=";".join(related_artist['genres']))
                    all_related_artists.add((current_artist_id, related_artist['id']))
                    if strategy == "BFS":
                        queue.append((related_artist['id'], current_artist_id))
                    else:
                        stack.append((related_artist['id'], current_artist_id))
            except Exception as e:
                print(f"Error processing artist {current_artist_id}: {e}")
                continue

    # Add edges for all related artists
    for parent_id, related_artist_id in all_related_artists:
        if parent_id in G and related_artist_id in G:
            G.add_edge(parent_id, related_artist_id)

    nx.write_graphml(G, out_filename)
    
    print(f"Total API calls made: {api_call_count}")  # Print the total number of API calls made
    return G



def get_track_data(sp, graph, output_file, batch_size=26, delay=10):
    """
    Get track data for each visited artist in the graph in batches to avoid rate limiting.

    :param sp: spotipy client object
    :param graph: graph with artists as nodes.
    :param output_file: name of the csv output file.
    :param batch_size: number of artists to process in each batch.
    :param delay: delay between batches in seconds.
    :return: pandas dataframe with track data.
    """
    tracks = []
    max_retries = 5

    # Helper function to handle retries and rate limits
    def fetch_with_retries(func, *args, **kwargs):
        retries = 0
        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except spotipy.exceptions.SpotifyException as e:
                if e.http_status == 429:
                    retry_after = int(e.headers.get('Retry-After', delay))
                    print(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                else:
                    retries += 1
                    print(f"Retrying ({retries}/{max_retries}) due to error: {e}")
                    time.sleep(5)  # Wait for 5 seconds before retrying
            except Exception as e:
                retries += 1
                print(f"Retrying ({retries}/{max_retries}) due to error: {e}")
                time.sleep(5)  # Wait for 5 seconds before retrying
        return None

    all_artist_ids = [node for node in graph.nodes]

    for i in range(0, len(all_artist_ids), batch_size):
        batch_artist_ids = all_artist_ids[i:i+batch_size]

        for artist_id in batch_artist_ids:
            results = fetch_with_retries(sp.artist_top_tracks, artist_id)
            if results is None:
                continue

            for track in results['tracks']:
                # Fetch audio features
                audio_features = fetch_with_retries(sp.audio_features, track['id'])
                if not audio_features or not audio_features[0]:
                    continue
                audio_features = audio_features[0]

                tracks.append({
                    'artist_id': artist_id,
                    'artist_name': track['artists'][0]['name'],
                    'track_id': track['id'],
                    'track_name': track['name'],
                    'track_duration': track['duration_ms'],
                    'track_popularity': track['popularity'],
                    'danceability': audio_features['danceability'],
                    'energy': audio_features['energy'],
                    'loudness': audio_features['loudness'],
                    'speechiness': audio_features['speechiness'],
                    'acousticness': audio_features['acousticness'],
                    'instrumentalness': audio_features['instrumentalness'],
                    'liveness': audio_features['liveness'],
                    'valence': audio_features['valence'],
                    'tempo': audio_features['tempo'],
                    'album_id': track['album']['id'],
                    'album_name': track['album']['name'],
                    'album_release_date': track['album']['release_date']
                })

        # Wait for the specified delay before processing the next batch
        if i + batch_size < len(all_artist_ids):
            print(f"Batch {i // batch_size + 1} processed. Waiting for {delay} seconds...")
            time.sleep(delay)

    # Save the data to a csv file so we can correctly output the songs.csv
    df = pd.DataFrame(tracks)
    df.to_csv(output_file, index=False)

    return df


if __name__ == "__main__":

    CLIENT_ID = ""
    CLIENT_SECRET = ""



    auth_manager = SpotifyClientCredentials(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    sp = spotipy.Spotify(auth_manager=auth_manager)

    id1 = search_artist(sp, "Taylor Swift")
    print("Taylor Swift id:", id1)

    Gb = crawler(sp, id1, 100, strategy='BFS', out_filename="gB.graphml") # 100 nodes/artists
    stats_gb = degree_statistics(Gb)
    print("Taylor Swift crawled BFS")
    print(f"gB degree stats: {stats_gb}")
    

    Gd = crawler(sp, id1, 100, strategy='DFS', out_filename="gD.graphml")
    print("Taylor Swift crawled DFS")
    stats_gd = degree_statistics(Gd)
    print(f"gD degree stats: {stats_gd}")

    id2 = search_artist(sp, "Pastel Ghost")
    print("Taylor Swift id:", id1)

    Hb = crawler(sp, id2, 100, strategy='BFS', out_filename="hB.graphml")
    print("Pastel Ghost crawled BFS")
    print(f"hB degree stats: {degree_statistics(Hb)}")



    # Read the graphs
    Gb = nx.read_graphml("gB.graphml")
    Gd = nx.read_graphml("gD.graphml")

    # Intersect the graphs
    G_intersection = nx.intersection(Gb, Gd)
    print(f"Number of nodes in the intersection: {G_intersection.number_of_nodes()}")
    print(f"Number of edges in the intersection: {G_intersection.number_of_edges()}")

    # Track data
    D = get_track_data(sp, G_intersection, "songs.csv")

    # Summary statistics
    num_songs = D['track_id'].nunique()
    num_artists = D['artist_id'].nunique()
    num_albums = D['album_id'].nunique()

    print(f"Artists: {num_artists}")
    print(f"Albums: {num_albums}")
    print(f"Songs: {num_songs}")
