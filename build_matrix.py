import pandas as pd
import osmnx as ox
import networkx as nx
import numpy as np

def build_matrix():
    print("1. Reading coordinates from austin_nodes.csv...")
    df = pd.read_csv('austin_nodes.csv')

    print("2. Downloading Austin road network from OpenStreetMap...")
    G = ox.graph_from_place('Austin, Texas, USA', network_type='drive')

    print("3. Imputing speed limits and edge travel times...")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Convert to strongly connected graph to ensure all nodes can reach each other
    G = ox.truncate.largest_component(G, strongly=True)

    print("4. Matching CSV coordinates to nearest road network nodes...")
    nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

    print("5. Calculating travel time matrix using Dijkstra's algorithm...")
    num_points = len(nodes)
    time_matrix = np.zeros((num_points, num_points), dtype=int)

    for i, origin_node in enumerate(nodes):
        lengths = nx.single_source_dijkstra_path_length(G, origin_node, weight='travel_time')
        for j, target_node in enumerate(nodes):
            if i != j:
                # If target_node is unreachable for any reason, default to a high penalty time (e.g. 99999)
                travel_time = lengths.get(target_node, 99999)
                time_matrix[i][j] = int(round(travel_time))

    print("6. Saving matrix to time_matrix.npy...")
    np.save('time_matrix.npy', time_matrix)
    print("\nSUCCESS! Travel time matrix generated and saved to 'time_matrix.npy'.")

if __name__ == '__main__':
    build_matrix()