import streamlit as st
import folium
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def generate_map(airport_network, name="airports"):
    # Add dots for each airport
    # Create base map (centered on the world)
    m = folium.Map(location=[20, 0], zoom_start=2)
    for node in airport_network.graph.nodes():
        folium.CircleMarker(
            location=[airport_network.graph.nodes[node]["latitude"], airport_network.graph.nodes[node]["longitude"]],
            radius=3,
            popup=folium.Popup(node_popup(airport_network.graph.nodes[node]), max_width=250),
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(m)


    # Add connections
    for edge in airport_network.graph.edges():
        source_node, target_node = edge
        source_data = airport_network.graph.nodes.get(source_node, {})
        target_data = airport_network.graph.nodes.get(target_node, {})
        
        if all(key in source_data for key in ['latitude', 'longitude']) and \
           all(key in target_data for key in ['latitude', 'longitude']):
            
            source_lat = source_data['latitude']
            source_lon = source_data['longitude']
            target_lat = target_data['latitude'] 
            target_lon = target_data['longitude']
            if all(coord is not None for coord in [source_lat, source_lon, target_lat, target_lon]):
                folium.PolyLine(
                    locations=[[source_lat, source_lon], [target_lat, target_lon]],
                    color='blue',
                    weight=1,
                    opacity=0.5,
                ).add_to(m)
    # Save map to HTML
    m.save(f"{name}_map.html")


def node_popup(source_node):
    return f"""
    <div style="font-family: Arial, sans-serif; width: 220px;">
        <p style="margin: 3px 0; font-size: 13px;"><b>Name:</b> {source_node.get("name", "")}</p>
        <p style="margin: 3px 0; font-size: 13px;"><b>City:</b> {source_node.get("city", "")}</p>
        <p style="margin: 3px 0; font-size: 13px;"><b>Country:</b> {source_node.get("country", "")}</p>
        <p style="margin: 3px 0; font-size: 13px;"><b>IATA Code:</b> {source_node.get("iata_code", "")}</p>
    </div>
    """


def regenerate_map(airport_network, name):
    """Regenerate and display the airport network map"""
    with st.spinner("Generating map..."):
        generate_map(airport_network, name=name)
    with open(f"{name}_map.html", "r") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)


def generate_map_with_path(airport_network, path_edges, start_node, end_node, name="path_map"):
    """Generate map with highlighted shortest path"""
    # Create base map (centered on the world)
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Convert path edges to set for fast lookup
    path_edges_set = set(path_edges)
    path_edges_reverse = set((edge[1], edge[0]) for edge in path_edges)
    all_path_edges = path_edges_set.union(path_edges_reverse)
    
    # Get all nodes in the path for highlighting
    path_nodes = set()
    for edge in path_edges:
        path_nodes.add(edge[0])
        path_nodes.add(edge[1])
    
    # Add dots for each airport
    for node in airport_network.graph.nodes():
        node_data = airport_network.graph.nodes[node]
        lat = node_data.get('latitude')
        lon = node_data.get('longitude')
        
        if lat is not None and lon is not None:
            # Create popup content
            popup_content = node_popup(node_data)
            
            # # Size based on degree (number of connections)
            degree = airport_network.graph.degree(node)
            size = min(max(degree / 2, 3), 15)

            # Highlight path nodes
            if node == start_node:
                color = 'green'
                fillColor = 'green'
                size = max(size, 10)
                popup_content = f"<b>START:</b> {popup_content}"
            elif node == end_node:
                color = 'red'
                fillColor = 'red'
                size = max(size, 10)
                popup_content = f"<b>END:</b> {popup_content}"
            elif node in path_nodes:
                color = 'purple'
                fillColor = 'purple'
                size = max(size, 8)
                popup_content = f"<b>PATH:</b> {popup_content}"
            else:
                color = 'blue'
                fillColor = 'blue'
                size = 3
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=size,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fillColor=fillColor,
                fillOpacity=0.8,
                weight=2
            ).add_to(m)

    # Add all regular connections first (background)
    for edge in airport_network.graph.edges():
        source_data = airport_network.graph.nodes[edge[0]]
        target_data = airport_network.graph.nodes[edge[1]]
        
        source_lat = source_data.get('latitude')
        source_lon = source_data.get('longitude')
        target_lat = target_data.get('latitude')
        target_lon = target_data.get('longitude')
        
        if all(coord is not None for coord in [source_lat, source_lon, target_lat, target_lon]):
            # Skip path edges (we'll draw them separately)
            if edge not in all_path_edges:
                # Check if bidirectional
                edge_data = airport_network.graph.edges[edge]
                is_bidirectional = edge_data.get('is_bidirectional', False)
                
                color = 'blue' if is_bidirectional else 'lightgreen'
                
                folium.PolyLine(
                    locations=[[source_lat, source_lon], [target_lat, target_lon]],
                    color=color,
                    weight=1,
                    opacity=0.3
                ).add_to(m)

    # Add highlighted path edges on top
    for i, (source, target) in enumerate(path_edges):
        source_data = airport_network.graph.nodes[source]
        target_data = airport_network.graph.nodes[target]
        
        source_lat = source_data.get('latitude')
        source_lon = source_data.get('longitude')
        target_lat = target_data.get('latitude')
        target_lon = target_data.get('longitude')
        
        if all(coord is not None for coord in [source_lat, source_lon, target_lat, target_lon]):
            # Create popup for the path segment
            segment_popup = f"""
            <b>Path Segment {i+1}</b><br>
            From: {source} ({source_data.get('name', 'Unknown')})<br>
            To: {target} ({target_data.get('name', 'Unknown')})
            """
            
            folium.PolyLine(
                locations=[[source_lat, source_lon], [target_lat, target_lon]],
                color='red',
                weight=4,
                opacity=0.9,
                popup=folium.Popup(segment_popup, max_width=250)
            ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Shortest Path Map</b></p>
    <p><i class="fa fa-circle" style="color:green"></i> Start Airport</p>
    <p><i class="fa fa-circle" style="color:red"></i> End Airport</p>
    <p><i class="fa fa-circle" style="color:purple"></i> Path Nodes</p>
    <p><span style="color:red; font-weight:bold;">━━━</span> Shortest Path</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map to HTML
    m.save(f"{name}_map.html")

def generate_map_with_size_based_on_degree(airport_network, name="degree_map"):
    """Generate map with airports sized by their degree (number of connections)"""
    # Create base map (centered on the world)
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Calculate degree statistics for sizing
    degrees = dict(airport_network.graph.degree())
    degree_values = list(degrees.values())
    
    if not degree_values:
        st.error("No degree data available")
        return
    
    min_degree = min(degree_values)
    max_degree = max(degree_values)
    
    # Create degree-based color and size mapping
    def get_degree_color_and_size(degree):
        # Size: scale between 3 and 20 based on degree
        if max_degree > min_degree:
            size = 3 + (degree - min_degree) * 17 / (max_degree - min_degree)
        else:
            size = 5
        
        # Color based on degree quartiles
        import numpy as np
        if degree >= np.percentile(degree_values, 90):
            return 'red', size, 'Very High Hub (Top 10%)'
        elif degree >= np.percentile(degree_values, 75):
            return 'orange', size, 'Major Hub (Top 25%)'
        elif degree >= np.percentile(degree_values, 50):
            return 'yellow', size, 'Medium Hub (Top 50%)'
        elif degree >= np.percentile(degree_values, 25):
            return 'lightgreen', size, 'Small Hub (Top 75%)'
        else:
            return 'lightblue', size, 'Regional Airport (Bottom 25%)'
    
    # Add dots for each airport with degree-based sizing and coloring
    for node in airport_network.graph.nodes():
        node_data = airport_network.graph.nodes[node]
        lat = node_data.get('latitude')
        lon = node_data.get('longitude')
        
        if lat is not None and lon is not None:
            degree = degrees.get(node, 0)
            color, size, category = get_degree_color_and_size(degree)
            
            # Enhanced popup with degree information
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 280px;">
                <h4 style="margin: 5px 0; color: {color};">{node_data.get("iata_code", node)}</h4>
                <p style="margin: 3px 0; font-size: 13px;"><b>Name:</b> {node_data.get("name", "Unknown")}</p>
                <p style="margin: 3px 0; font-size: 13px;"><b>City:</b> {node_data.get("city", "Unknown")}</p>
                <p style="margin: 3px 0; font-size: 13px;"><b>Country:</b> {node_data.get("country", "Unknown")}</p>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0; font-size: 16px; color: {color};"><b>🔗 Connections: {degree}</b></p>
                <p style="margin: 3px 0; font-size: 12px;"><b>Category:</b> {category}</p>
                <p style="margin: 3px 0; font-size: 12px;"><b>Hub Rank:</b> {sorted(degree_values, reverse=True).index(degree) + 1} of {len(degree_values)}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=size,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)

    # Create comprehensive legend with statistics
    import numpy as np
    avg_degree = np.mean(degree_values)
    median_degree = np.median(degree_values)
    
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 300px; height: 400px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; overflow-y: auto;">
    <h3 style="margin-top: 0;">🔗 Airport Connectivity Map</h3>
    
    <h4>Airport Categories:</h4>
    <p><i class="fa fa-circle" style="color:red; font-size:16px;"></i> Very High Hub (Top 10%): {np.percentile(degree_values, 90):.0f}+ connections</p>
    <p><i class="fa fa-circle" style="color:orange; font-size:14px;"></i> Major Hub (Top 25%): {np.percentile(degree_values, 75):.0f}+ connections</p>
    <p><i class="fa fa-circle" style="color:yellow; font-size:12px;"></i> Medium Hub (Top 50%): {np.percentile(degree_values, 50):.0f}+ connections</p>
    <p><i class="fa fa-circle" style="color:lightgreen; font-size:10px;"></i> Small Hub: {np.percentile(degree_values, 25):.0f}+ connections</p>
    <p><i class="fa fa-circle" style="color:lightblue; font-size:8px;"></i> Regional: < {np.percentile(degree_values, 25):.0f} connections</p>
    
    <h4>💡 How to Read:</h4>
    <p>• <b>Circle size</b> = Number of connections</p>
    <p>• <b>Color</b> = Hub category</p>
    <p>• <b>Click</b> airports for detailed info</p>
    <p>• <b>Larger circles</b> = Major aviation hubs</p>
    
    <small style="margin-top: 10px; display: block; font-style: italic;">
    This map shows airport importance based on connectivity only (no routes displayed)
    </small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map to HTML
    m.save(f"{name}_map.html")
    
    # Return statistics for further analysis
    return {
        'min_degree': min_degree,
        'max_degree': max_degree,
        'avg_degree': avg_degree,
        'median_degree': median_degree,
        'total_airports': len(airport_network.graph.nodes()),
        'degree_distribution': {
            'very_high': len([d for d in degree_values if d >= np.percentile(degree_values, 90)]),
            'major': len([d for d in degree_values if d >= np.percentile(degree_values, 75) and d < np.percentile(degree_values, 90)]),
            'medium': len([d for d in degree_values if d >= np.percentile(degree_values, 50) and d < np.percentile(degree_values, 75)]),
            'small': len([d for d in degree_values if d >= np.percentile(degree_values, 25) and d < np.percentile(degree_values, 50)]),
            'regional': len([d for d in degree_values if d < np.percentile(degree_values, 25)])
        }
    }


def generate_community_map(airport_network, communities, method_name, name="community_map"):
    """Generate map with communities highlighted in different colors"""
    # Create base map (centered on the world)
    m = folium.Map(location=[20, 0], zoom_start=2)
    
    # Get unique community IDs and create color palette
    unique_communities = list(set(communities.values()))
    num_communities = len(unique_communities)
    
    # Create color palette
    if num_communities <= 10:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    else:
        # For more communities, use matplotlib colormap
        cmap = cm.get_cmap('tab20')  # or 'Set3', 'tab10'
        colors = [mcolors.rgb2hex(cmap(i / num_communities)) for i in range(num_communities)]
    
    # Create community color mapping
    community_colors = {}
    for i, comm_id in enumerate(unique_communities):
        community_colors[comm_id] = colors[i % len(colors)]
    
    # Add dots for each airport with community-based coloring
    for node in airport_network.graph.nodes():
        node_data = airport_network.graph.nodes[node]
        lat = node_data.get('latitude')
        lon = node_data.get('longitude')
        
        if lat is not None and lon is not None:
            # Get community info
            community_id = communities.get(node, -1)
            color = community_colors.get(community_id, 'gray')
            
            # Size based on degree
            degree = airport_network.graph.degree(node)
            size = min(max(degree / 3, 4), 12)
            
            # Enhanced popup with community information
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; width: 250px;">
                <h4 style="margin: 5px 0; color: {color};">{node_data.get("iata_code", node)}</h4>
                <p style="margin: 3px 0; font-size: 13px;"><b>Name:</b> {node_data.get("name", "Unknown")}</p>
                <p style="margin: 3px 0; font-size: 13px;"><b>City:</b> {node_data.get("city", "Unknown")}</p>
                <p style="margin: 3px 0; font-size: 13px;"><b>Country:</b> {node_data.get("country", "Unknown")}</p>
                <hr style="margin: 5px 0;">
                <p style="margin: 3px 0; font-size: 14px; color: {color};"><b>🏘️ Community: {community_id + 1}</b></p>
                <p style="margin: 3px 0; font-size: 12px;"><b>🔗 Connections:</b> {degree}</p>
                <p style="margin: 3px 0; font-size: 12px;"><b>📊 Method:</b> {method_name}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=size,
                popup=folium.Popup(popup_content, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.8,
                weight=2
            ).add_to(m)

    # Add connections with community-aware styling
    for edge in airport_network.graph.edges():
        source_data = airport_network.graph.nodes[edge[0]]
        target_data = airport_network.graph.nodes[edge[1]]
        
        source_lat = source_data.get('latitude')
        source_lon = source_data.get('longitude')
        target_lat = target_data.get('latitude')
        target_lon = target_data.get('longitude')
        
        if all(coord is not None for coord in [source_lat, source_lon, target_lat, target_lon]):
            # Get community IDs for both endpoints
            source_community = communities.get(edge[0], -1)
            target_community = communities.get(edge[1], -1)
            
            # Style based on whether connection is within or between communities
            if source_community == target_community and source_community != -1:
                # Intra-community connection
                color = community_colors.get(source_community, 'gray')
                weight = 2
                opacity = 0.7
                dash_array = None
            else:
                # Inter-community connection
                color = 'black'
                weight = 1
                opacity = 0.3
                dash_array = '5, 5'  # Dashed line
            
            folium.PolyLine(
                locations=[[source_lat, source_lon], [target_lat, target_lon]],
                color=color,
                weight=weight,
                opacity=opacity,
                dash_array=dash_array
            ).add_to(m)

    # Create comprehensive legend
    legend_items = []
    community_stats = {}
    
    # Calculate community statistics
    for comm_id in unique_communities:
        community_nodes = [node for node, c_id in communities.items() if c_id == comm_id]
        community_stats[comm_id] = {
            'size': len(community_nodes),
            'color': community_colors[comm_id]
        }
    
    # Sort communities by size for legend
    sorted_communities = sorted(community_stats.items(), key=lambda x: x[1]['size'], reverse=True)
    
    legend_html = f'''
    <div style="position: fixed; 
                top: 10px; right: 10px; width: 280px; height: 450px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:12px; padding: 15px; overflow-y: auto;">
    <h3 style="margin-top: 0;">🏘️ Community Detection</h3>
    <p><b>Method:</b> {method_name}</p>
    <p><b>Total Communities:</b> {num_communities}</p>
    <p><b>Total Airports:</b> {len(airport_network.graph.nodes())}</p>
    
    <h4>🎨 Communities (by size):</h4>
    '''
    
    # Add top 10 communities to legend
    for i, (comm_id, stats) in enumerate(sorted_communities[:10]):
        legend_html += f'<p><i class="fa fa-circle" style="color:{stats["color"]}"></i> Community {comm_id + 1}: {stats["size"]} airports</p>'
    
    if len(sorted_communities) > 10:
        remaining = len(sorted_communities) - 10
        legend_html += f'<p>... and {remaining} more communities</p>'
    
    legend_html += '''
    <h4>🔗 Connection Types:</h4>
    <p><span style="color:colored; font-weight:bold;">━━━</span> Intra-community</p>
    <p><span style="color:black;">╌╌╌</span> Inter-community</p>
    
    <small style="margin-top: 10px; display: block;">
    💡 Circle size = degree centrality<br>
    Same colors = same community<br>
    Click airports for details
    </small>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))

    # Save map to HTML
    m.save(f"{name}_map.html")
    
    return community_stats