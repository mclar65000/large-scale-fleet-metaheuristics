import pandas as pd
import random
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

#load in CSV and Time Matrix
df = pd.read_csv('austin_nodes.csv')
time_matrix = np.load('time_matrix.npy').tolist()

# generate demands (0 for depot, 1-5 units for stops)
import random
random.seed(42) # optional: keep demands consistent
demands = [0] + [random.randint(1, 5) for _ in range(len(df) - 1)]

num_locations = len(df)
num_vehicles = 5
depot_index = 0

#load Road Network Graph for Street-Level Plotting
print("Loading Austin road network for exact street route tracing...")
G = ox.graph_from_place('Austin, Texas, USA', network_type='drive')
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)
G = ox.truncate.largest_component(G, strongly=True)

#map CSV coordinates to the nearest street graph nodes
osm_nodes = ox.nearest_nodes(G, X=df['longitude'], Y=df['latitude'])

#initialize Routing Engine
manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, depot_index)
routing = pywrapcp.RoutingModel(manager)

#use all vehicles
solver = routing.solver()
for vehicle_id in range(num_vehicles):
    solver.Add(routing.ActiveVehicleVar(vehicle_id) == 1)

service_times = df['service_time_sec'].tolist()

#register transit callback
transit_callback_index = routing.RegisterTransitCallback(
    lambda from_idx, to_idx: time_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] + service_times[manager.IndexToNode(from_idx)]
)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

#add Time Window Constraints
time_dimension_name = 'Time'
routing.AddDimension(
    transit_callback_index,
    3600,   #waiting time allowed (1 hour)
    28800,  #max vehicle shift duration (8 hours)
    False,  #do not force start cumulative time to zero
    time_dimension_name
)
time_dimension = routing.GetDimensionOrDie(time_dimension_name)

#add capacity constraints
demand_callback_index = routing.RegisterUnaryTransitCallback(
    lambda from_idx: demands[manager.IndexToNode(from_idx)]
)
vehicle_capacity = 50 #max capacity per vehicle
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,  #null capacity slack
    [vehicle_capacity] * num_vehicles,  #vehicle max capacities
    True,  #start cumul to zero
    'Capacity'
)

#set time windows per location from CSV
for location_idx, row in df.iterrows():
    index = manager.NodeToIndex(location_idx)
    time_dimension.CumulVar(index).SetRange(
        int(row['tw_open_sec']), 
        int(row['tw_close_sec'])
    )

#minimize shift duration
for i in range(num_vehicles):
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(i)))
    routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(i)))

#search parameters and guided local search metaheuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30

#solve
print("Solving VRPTW with Google OR-Tools...")
solution = routing.SolveWithParameters(search_parameters)

#===============================================
#print Schedules & Export Interactive Street Map
#===============================================

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
        vehicle_stops = []
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            vehicle_stops.append(node)
            time_var = time_dimension.CumulVar(index)
            current_time = solution.Value(time_var)
            if node == depot_index:
                plan_output += f"  -> Depot start | Time: {current_time}s\n"
            else:
                plan_output += f"  -> Stop {node:02d} ({df.loc[node, 'name']}) | Arrival: {current_time}s\n"

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            next_time = solution.Value(time_dimension.CumulVar(next_index))
            drive_time = time_matrix[node][next_node]
            wait_time = max(
                0,
                next_time - current_time - drive_time - service_times[node]
            )
            plan_output += (
                f"       Leg to {next_node:02d}: drive {drive_time}s, "
                f"wait at {node:02d} {wait_time}s\n"
            )
            
            lat, lon = df.loc[node, 'latitude'], df.loc[node, 'longitude']
            
            #place Marker for Stop
            folium.Marker(
                location=[lat, lon],
                popup=f"V{vehicle_id + 1} - Stop {node}: {df.loc[node, 'name']}",
                icon=folium.Icon(color=colors[vehicle_id % len(colors)])
            ).add_to(map_austin)
            
            index = next_index

        #add return to depot stop
        end_node = manager.IndexToNode(index)
        vehicle_stops.append(end_node)
        time_var = time_dimension.CumulVar(index)
        plan_output += f"  -> Return Depot ({df.loc[end_node, 'name']}) | Arrival: {solution.Min(time_var)}s\n"
        print(plan_output)

        #tracing exact street geometry node-by-node along the route
        for i in range(len(vehicle_stops) - 1):
            orig_osm = osm_nodes[vehicle_stops[i]]
            dest_osm = osm_nodes[vehicle_stops[i + 1]]
            
            try:
                #find shortest path along actual road segments
                path = nx.shortest_path(G, orig_osm, dest_osm, weight='travel_time')
                street_coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in path]
                
                folium.PolyLine(
                    street_coords, 
                    color=colors[vehicle_id % len(colors)], 
                    weight=4, 
                    opacity=0.8,
                    tooltip=f"Vehicle {vehicle_id + 1} Route"
                ).add_to(map_austin)
            except nx.NetworkXNoPath:
                #fallback to straight line if path lookup fails
                loc1 = (df.loc[vehicle_stops[i], 'latitude'], df.loc[vehicle_stops[i], 'longitude'])
                loc2 = (df.loc[vehicle_stops[i + 1], 'latitude'], df.loc[vehicle_stops[i + 1], 'longitude'])
                folium.PolyLine([loc1, loc2], color=colors[vehicle_id % len(colors)], weight=3, opacity=0.5).add_to(map_austin)

    map_austin.save('route_map.html')
    print("\nSUCCESS! Saved interactive street-level route map to 'route_map.html'.")
else:
    print("No solution found. Consider increasing vehicle count or relaxing time windows.")