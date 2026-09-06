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

# 7. Print Schedules & Export Map
if solution:
    print("\n=== OPTIMAL ROUTES FOUND ===")
    map_austin = folium.Map(
        location=[df.loc[depot_index, 'latitude'], df.loc[depot_index, 'longitude']],
        zoom_start=12
    )
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue']

    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        plan_output = f"\nVehicle {vehicle_id + 1}:\n"
        route_coords = []
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            plan_output += f"  -> Stop {node:02d} ({df.loc[node, 'name']}) | Arrival: {solution.Min(time_var)}s\n"
            
            lat, lon = df.loc[node, 'latitude'], df.loc[node, 'longitude']
            route_coords.append((lat, lon))
            
            folium.Marker(
                location=[lat, lon],
                popup=f"V{vehicle_id + 1} - Stop {node}: {df.loc[node, 'name']}",
                icon=folium.Icon(color=colors[vehicle_id % len(colors)])
            ).add_to(map_austin)
            
            index = solution.Value(routing.NextVar(index))

        # Depot Return
        node = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        plan_output += f"  -> Return Depot ({df.loc[node, 'name']}) | Arrival: {solution.Min(time_var)}s\n"
        route_coords.append((df.loc[node, 'latitude'], df.loc[node, 'longitude']))
        
        print(plan_output)

        # Draw Route Line
        folium.PolyLine(
            route_coords, 
            color=colors[vehicle_id % len(colors)], 
            weight=3, 
            opacity=0.8,
            tooltip=f"Vehicle {vehicle_id + 1} Route"
        ).add_to(map_austin)

    map_austin.save('route_map.html')
    print("\nSUCCESS! Saved interactive route map to 'route_map.html'.")
else:
    print("No solution found. Consider increasing vehicle count or relaxing time windows.")

