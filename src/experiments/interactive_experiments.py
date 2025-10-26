import streamlit as st
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from collections import defaultdict
import numpy as np
from src.components.map_generator import regenerate_map, generate_map
from src.AirportNetwork import AirportNetwork
from src.components.map_generator import (
    generate_map_with_path, 
    generate_community_map
)


class Experiment():
    def __init__(self, airport_network):
        self.airport_network = airport_network
        self.graph_per_experiment = AirportNetwork.create_from_existing(airport_network)

    def display_network_experiments(self):
        """Display interactive experiments for network analysis"""
        
        experiment_type = st.selectbox(
            "Choose an experiment:",
            [
                "🎯 Hub Removal Impact",
                "🛤️ Shortest Path Finder", 
                "🔍 Community Detection",
                # "📈 Growth Simulation",
                "🎲 Random Walk Simulation",
            ],
            key="experiment_type"
        )
        
        if experiment_type == "🎯 Hub Removal Impact":
            self.hub_removal_experiment()
            regenerate_map(self.airport_network, name="hub_removal_map")
        elif experiment_type == "🛤️ Shortest Path Finder":
            self.shortest_path_experiment()
            regenerate_map(self.airport_network, name="shortest_path_map")
        elif experiment_type == "🔍 Community Detection":
            self.community_detection_experiment()
        elif experiment_type == "🎲 Random Walk Simulation":
            self.random_walk_experiment()

    def hub_removal_experiment(self):
        """Experiment: What happens when we remove major hubs?"""
        st.subheader("🎯 Hub Removal Impact Analysis")
        st.write("**Question:** How does removing major hubs affect network connectivity?")
        
        # Get top hubs by degree
        degrees = dict(self.graph_per_experiment.graph.degree())
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # User selects hub to remove
        selected_hub = st.selectbox(
            "Select a hub to remove:",
            options=[f"{hub[0]} ({hub[1]} connections)" for hub in top_hubs],
            help="Choose a major hub to see impact on network",
            key="hub_removal_select"
        )
        
        hub_code = selected_hub.split()[0]

        if st.button("🔥 Remove Hub and Analyze Impact"):
            # Create copy without the hub
            original_stats = {
                'nodes': self.airport_network.graph.number_of_nodes(),
                'edges': self.airport_network.graph.number_of_edges(),
                'components': nx.number_weakly_connected_components(self.airport_network.graph),
                'largest_component': len(max(nx.weakly_connected_components(self.airport_network.graph), key=len))
            }
            
            # Remove hub
            self.graph_per_experiment.graph.remove_node(hub_code)

            new_stats = {
                'nodes': self.graph_per_experiment.graph.number_of_nodes(),
                'edges': self.graph_per_experiment.graph.number_of_edges(),
                'components': nx.number_weakly_connected_components(self.graph_per_experiment.graph),
                'largest_component': len(max(nx.weakly_connected_components(self.graph_per_experiment.graph), key=len)) if self.graph_per_experiment.graph.number_of_nodes() > 0 else 0
            }
            
            # Display impact
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Airports", new_stats['nodes'], delta=new_stats['nodes'] - original_stats['nodes'])
            with col2:
                st.metric("Routes", new_stats['edges'], delta=new_stats['edges'] - original_stats['edges'])
            with col3:
                st.metric("Components", new_stats['components'], delta=new_stats['components'] - original_stats['components'])
            with col4:
                st.metric("Largest Component", new_stats['largest_component'], 
                        delta=new_stats['largest_component'] - original_stats['largest_component'])
            
            st.success(f"✅ Removed {hub_code} - Impact analysis complete!")
            regenerate_map(self.graph_per_experiment, name="hub_removal_map")

    def generate_path_highlighted_map(self, path, start_airport, end_airport):
        """Generate map with shortest path highlighted in red"""
        try:
            # Create path edges for highlighting
            path_edges = []
            for i in range(len(path) - 1):
                path_edges.append((path[i], path[i + 1]))
            
            # Use the new path highlighting function
            generate_map_with_path(
                self.airport_network, 
                path_edges=path_edges,
                start_node=start_airport,
                end_node=end_airport,
                name="shortest_path_highlighted"
            )
            
            # Display the map
            with open("shortest_path_highlighted_map.html", "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=500, scrolling=True)
            
        except Exception as e:
            st.error(f"❌ Could not generate path map: {str(e)}")
            # Fallback to regular map
            regenerate_map(self.airport_network, name="shortest_path_fallback")

    def shortest_path_experiment(self):
        """Experiment: Find shortest paths between airports"""
        st.subheader("🛤️ Shortest Path Finder")
        st.write("**Question:** What's the shortest route between any two airports?")
        
        airports = list(self.airport_network.graph.nodes())
        
        col1, col2 = st.columns(2)
        with col1:
            start_airport = st.selectbox("From:", airports, key="start")
        with col2:
            end_airport = st.selectbox("To:", airports, key="end")
        
        if st.button("🔍 Find Shortest Path"):
            try:
                undirected_graph = self.airport_network.graph.to_undirected()
                path = nx.shortest_path(undirected_graph, start_airport, end_airport)
                path_length = len(path) - 1
                
                st.success(f"✅ Shortest path found! **{path_length} stops**")
                
                # Display path
                st.write("**Route:**")
                path_details = []
                for i, airport in enumerate(path):
                    airport_data = self.airport_network.graph.nodes.get(airport, {})
                    path_details.append({
                        'Step': i + 1,
                        'Airport': airport,
                        'Name': airport_data.get('name', 'Unknown'),
                        'City': airport_data.get('city', 'Unknown'),
                        'Country': airport_data.get('country', 'Unknown')
                    })
                
                st.dataframe(pd.DataFrame(path_details), use_container_width=True)

                # Generate map with highlighted path
                self.generate_path_highlighted_map(path, start_airport, end_airport)
                # Alternative paths
                try:
                    all_paths = list(nx.all_shortest_paths(undirected_graph, start_airport, end_airport))
                    if len(all_paths) > 1:
                        st.info(f"Found {len(all_paths)} alternative routes with the same length")
                    regenerate_map(self.airport_network, name="shortest_path_map")
                except:
                    pass
                    
            except nx.NetworkXNoPath:
                st.error("❌ No path exists between these airports!")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    def community_detection_experiment(self):
        """Experiment: Detect communities in the network"""
        st.subheader("🔍 Community Detection")
        st.write("**Question:** Can we identify natural clusters of airports?")
        
        method = st.selectbox(
            "Choose detection method:",
            ["Greedy Modularity", "K-Core Decomposition"]
        )
        
        if st.button("🔍 Detect Communities"):
            undirected_graph = self.airport_network.graph.to_undirected()

            if method == "Greedy Modularity":
                communities_generator = nx.community.greedy_modularity_communities(undirected_graph)
                communities = {}
                for i, community in enumerate(communities_generator):
                    for node in community:
                        communities[node] = i
                        
            else:  # K-Core
                communities = nx.core_number(undirected_graph)

            community_stats = generate_community_map(
                self.airport_network, 
                communities, 
                method, 
                name="community_detection"
            )

            with open(f"community_detection_map.html", "r") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600, scrolling=True)
            # Analyze communities
            community_stats = defaultdict(list)
            for node, community_id in communities.items():
                node_data = self.airport_network.graph.nodes.get(node, {})
                community_stats[community_id].append({
                    'airport': node,
                    'country': node_data.get('country', 'Unknown')
                })
            
            st.success(f"✅ Found {len(community_stats)} communities!")
            
            # Display largest communities
            largest_communities = sorted(community_stats.items(), 
                                    key=lambda x: len(x[1]), reverse=True)[:5]
            
            for i, (comm_id, airports) in enumerate(largest_communities):
                with st.expander(f"Community {comm_id + 1} ({len(airports)} airports)"):
                    countries = [a['country'] for a in airports]
                    country_counts = pd.Series(countries).value_counts()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Top Countries:**")
                        st.write(country_counts.head())
                    with col2:
                        st.write("**Sample Airports:**")
                        sample = airports[:10]
                        for airport in sample:
                            st.write(f"• {airport['airport']} ({airport['country']})")


    def random_walk_experiment(self):
        """Experiment: Simulate random walks through the network"""
        st.subheader("🎲 Random Walk Simulation")
        st.write("**Question:** Where do random flights lead us?")

        start_airport = st.selectbox("Starting airport:", list(self.airport_network.graph.nodes()))
        walk_length = st.slider("Walk length:", 5, 50, 20)
        num_walks = st.slider("Number of walks:", 1, 100, 10)
        
        if st.button("🚶 Start Random Walk"):
            all_visits = defaultdict(int)
            walk_paths = []
            
            for walk_num in range(num_walks):
                current = start_airport
                path = [current]
                
                for _ in range(walk_length):
                    neighbors = list(self.airport_network.graph.neighbors(current))
                    if neighbors:
                        current = random.choice(neighbors)
                        path.append(current)
                        all_visits[current] += 1
                    else:
                        break
                
                walk_paths.append(path)
            
            # Most visited airports
            most_visited = sorted(all_visits.items(), key=lambda x: x[1], reverse=True)[:10]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Most Visited Airports:**")
                visit_data = []
                for airport, visits in most_visited:
                    airport_data = self.airport_network.graph.nodes.get(airport, {})
                    visit_data.append({
                        'Airport': airport,
                        'Name': airport_data.get('name', 'Unknown'),
                        'Visits': visits,
                        'Percentage': f"{visits/(num_walks*walk_length)*100:.1f}%"
                    })
                st.dataframe(pd.DataFrame(visit_data), use_container_width=True)
            
            with col2:
                st.write("**Sample Walk Path:**")
                sample_path = walk_paths[0][:10]  # Show first 10 steps
                for i, airport in enumerate(sample_path):
                    airport_data = self.airport_network.graph.nodes.get(airport, {})
                    st.write(f"{i+1}. {airport} - {airport_data.get('name', 'Unknown')}")
