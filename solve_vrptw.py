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

# Inline lambda callback required by OR-Tools
transit_callback_index = routing.RegisterTransitCallback(
    lambda from_idx, to_idx: time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] + service_times[manager.IndexToNode(from_idx)]
)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# 4. Add Time Window Constraints
time_dimension_name = 'Time'
routing.AddDimension(
    transit_callback_index,
    3600,   # Allow up to 1 hour (3600s) of waiting time if a vehicle arrives early
    28800,  # Max vehicle shift duration (8 hours = 28800s)
    False,  # do not force cumulative time to zero at start
    time_dimension_name
)
time_dimension = routing.GetDimensionOrDie(time_dimension_name)

# Set individual open/close time windows from CSV
for location_idx, row in df.iterrows():
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(
        int(row['tw_open_sec']), 
        int(row['tw_close_sec'])
    )

# Minimize total operational shift time across vehicles
for i in range(num_vehicles):
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

# 5. Search Parameters & Guided Local Search Metaheuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30

# 6. Solve
print("Solving VRPTW with Google OR-Tools...")
solution = routing.SolveWithParameters(search_parameters)

