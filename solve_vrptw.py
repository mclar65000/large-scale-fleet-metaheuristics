import pandas as pd
import numpy as np
import folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# 1. Load Input Data
df = pd.read_csv('austin_nodes.csv')
time_matrix = np.load('time_matrix.npy').tolist()

num_locations = len(df)
num_vehicles = 5
depot_index = 0

# 2. Initialize Routing Engine
manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
routing = pywrapcp.RoutingModel(manager)

# 3. Travel Time Callback (Travel Time + Service Duration)
service_times = df['service_time_sec'].tolist()


