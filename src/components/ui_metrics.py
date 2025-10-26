import yaml
from src.utils import metrics
import streamlit as st
import networkx as nx
import polars as pl
import plotly.express as px
import numpy as np
import scipy.stats as stats
from src.components.map_generator import generate_map_with_size_based_on_degree


def generate_countries_list():
    # Load countries
    with open("data/countries.yaml", 'r', encoding='utf-8') as file:
        countries_data = yaml.safe_load(file)

    countries_list = countries_data['countries'] 

    # Create country filter in sidebar
    st.sidebar.header("🌍 Country Selection")
    use_filter = st.sidebar.checkbox("Filter by specific countries", value=True)

    if use_filter:
        selected_countries = st.sidebar.multiselect(
            "Select Countries",
            options=countries_list,
            default=["United States", "Canada", "United Kingdom"]
        )
            
        st.sidebar.success(f"Selected {len(selected_countries)} countries")
        return selected_countries
    else:
        st.sidebar.info("Using all countries (global network)")
        return None  


def metrics_display(airport_network):
    metrics = {}
    generate_map_with_size_based_on_degree(airport_network, "degree_map")
    with open(f"degree_map_map.html", "r") as f:
        html_content = f.read()
    st.components.v1.html(html_content, height=600, scrolling=True)
    basic_network_overview(metrics, airport_network)
    network_structure_analysis(metrics, airport_network)
    centrality_metrics(metrics, airport_network)


def basic_network_overview(metrics, airport_network):
    """Display basic network statistics"""
    st.subheader("Basic Network Overview")
    
    # Calculate basic metrics
    metrics['nodes'] = airport_network.graph.number_of_nodes()
    metrics['edges'] = airport_network.graph.number_of_edges()
    metrics['density'] = nx.density(airport_network.graph)
    
    # Calculate additional metrics
    try:
        metrics['avg_degree'] = sum(dict(airport_network.graph.degree()).values()) / metrics['nodes']
        metrics['avg_path_length'] = nx.average_shortest_path_length(airport_network.graph.to_undirected()) if nx.is_connected(airport_network.graph.to_undirected()) else "N/A"
    except:
        metrics['avg_degree'] = "N/A"
        metrics['diameter'] = "N/A"
        metrics['avg_path_length'] = "N/A"
    
    # Display in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Airports", metrics['nodes'])
        st.metric("Average Degree", f"{metrics['avg_degree']:.2f}" if isinstance(metrics['avg_degree'], float) else metrics['avg_degree'])
    
    with col2:
        st.metric("Total Routes", metrics['edges'])
        st.metric("Network Density", f"{metrics['density']:.4f}")
    
    with col3:
        # Calculate connectivity stats
        largest_component_size = len(max(nx.weakly_connected_components(airport_network.graph), key=len))
        st.metric("Largest Component", f"{largest_component_size}")
        connected_components = nx.number_weakly_connected_components(airport_network.graph)
        st.metric("Connected Components", connected_components)
        


def centrality_metrics(metrics, airport_network):
    """Display detailed centrality measures analysis"""
    st.subheader("🎯 Node Centrality Measures")
    
    st.write("**Analysis of the most important airports by different centrality measures:**")
    
    # Sample for large graphs to improve performance
    sample_size = min(1000, metrics['nodes'])
    
    with st.spinner("Calculating centrality measures..."):
        # Calculate all centrality measures
        degree_centrality = nx.degree_centrality(airport_network.graph)
        betweenness_centrality = nx.betweenness_centrality(airport_network.graph, k=sample_size)
        closeness_centrality = nx.closeness_centrality(airport_network.graph)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(airport_network.graph, max_iter=1000)
        except:
            eigenvector_centrality = {}
            st.warning("⚠️ Could not calculate eigenvector centrality (graph may not be strongly connected)")

    # Top nodes by centrality
    metrics['top_degree'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics['top_betweenness'] = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics['top_closeness'] = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    metrics['top_eigenvector'] = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10] if eigenvector_centrality else []

    # Display centrality results in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔗 Top 10 - Degree Centrality** (Most Connected)")
        st.caption("Measures the number of direct connections an airport has")
        
        degree_data = []
        for i, (airport, centrality) in enumerate(metrics['top_degree']):
            airport_data = airport_network.graph.nodes.get(airport, {})
            degree = airport_network.graph.degree(airport)
            degree_data.append({
                'Rank': i + 1,
                'Code': airport,
                'Name': airport_data.get('name', 'Unknown')[:30],
                'Country': airport_data.get('country', 'Unknown'),
                'Connections': degree,
                'Centrality': f"{centrality:.4f}"
            })
        
        df_degree = pl.DataFrame(degree_data)
        st.dataframe(df_degree, use_container_width=True, hide_index=True)
        
        st.write("**🎯 Top 10 - Closeness Centrality** (Most Central)")
        st.caption("Measures how close an airport is to all other airports")
        
        closeness_data = []
        for i, (airport, centrality) in enumerate(metrics['top_closeness']):
            airport_data = airport_network.graph.nodes.get(airport, {})
            closeness_data.append({
                'Rank': i + 1,
                'Code': airport,
                'Name': airport_data.get('name', 'Unknown')[:30],
                'Country': airport_data.get('country', 'Unknown'),
                'Centrality': f"{centrality:.4f}"
            })
        
        df_closeness = pl.DataFrame(closeness_data)
        st.dataframe(df_closeness, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**🌉 Top 10 - Betweenness Centrality** (Most Critical)")
        st.caption("Measures how often an airport lies on shortest paths between others")
        
        betweenness_data = []
        for i, (airport, centrality) in enumerate(metrics['top_betweenness']):
            airport_data = airport_network.graph.nodes.get(airport, {})
            betweenness_data.append({
                'Rank': i + 1,
                'Code': airport,
                'Name': airport_data.get('name', 'Unknown')[:30],
                'Country': airport_data.get('country', 'Unknown'),
                'Centrality': f"{centrality:.4f}"
            })
        
        df_betweenness = pl.DataFrame(betweenness_data)
        st.dataframe(df_betweenness, use_container_width=True, hide_index=True)
        
        if metrics['top_eigenvector']:
            st.write("**⭐ Top 10 - Eigenvector Centrality** (Most Influential)")
            st.caption("Measures connections to other important airports")
            
            eigenvector_data = []
            for i, (airport, centrality) in enumerate(metrics['top_eigenvector']):
                airport_data = airport_network.graph.nodes.get(airport, {})
                eigenvector_data.append({
                    'Rank': i + 1,
                    'Code': airport,
                    'Name': airport_data.get('name', 'Unknown')[:30],
                    'Country': airport_data.get('country', 'Unknown'),
                    'Centrality': f"{centrality:.4f}"
                })
            
            df_eigenvector = pl.DataFrame(eigenvector_data)
            st.dataframe(df_eigenvector, use_container_width=True, hide_index=True)
        else:
            st.info("🔍 Eigenvector centrality not available for this graph")

    # Comparative centrality analysis
    st.subheader("📈 Comparative Centrality Analysis")
    
    # Check which airports appear in multiple rankings
    all_top_airports = set()
    ranking_appearances = {}
    
    for airport, _ in metrics['top_degree']:
        all_top_airports.add(airport)
        ranking_appearances[airport] = ranking_appearances.get(airport, 0) + 1
    
    for airport, _ in metrics['top_betweenness']:
        all_top_airports.add(airport)
        ranking_appearances[airport] = ranking_appearances.get(airport, 0) + 1
    
    for airport, _ in metrics['top_closeness']:
        all_top_airports.add(airport)
        ranking_appearances[airport] = ranking_appearances.get(airport, 0) + 1
    
    if metrics['top_eigenvector']:
        for airport, _ in metrics['top_eigenvector']:
            all_top_airports.add(airport)
            ranking_appearances[airport] = ranking_appearances.get(airport, 0) + 1
    
    # Airports that appear in multiple rankings
    multi_ranking = [(airport, count) for airport, count in ranking_appearances.items() if count >= 3]
    multi_ranking.sort(key=lambda x: x[1], reverse=True)
    
    if multi_ranking:
        st.write("**🏆 Super-Hubs (appear in ≥3 rankings):**")
        
        superhub_data = []
        for airport, appearances in multi_ranking:
            airport_data = airport_network.graph.nodes.get(airport, {})
            degree = airport_network.graph.degree(airport)
            superhub_data.append({
                'Code': airport,
                'Name': airport_data.get('name', 'Unknown'),
                'Country': airport_data.get('country', 'Unknown'),
                'Connections': degree,
                'Rankings': f"{appearances}/4",
                'Status': '🌟 Super-Hub' if appearances == 4 else '⭐ Multi-Hub'
            })
        
        df_superhubs = pl.DataFrame(superhub_data)
        st.dataframe(df_superhubs, use_container_width=True, hide_index=True)
    
    # Centrality distribution visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Degree Centrality Distribution**")
        degree_values = list(degree_centrality.values())
        
        fig = px.histogram(
            x=degree_values,
            nbins=20,
            title="Degree Centrality Distribution",
            labels={'x': 'Degree Centrality', 'y': 'Number of Airports'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**📊 Betweenness Centrality Distribution**")
        betweenness_values = list(betweenness_centrality.values())
        
        fig = px.histogram(
            x=betweenness_values,
            nbins=20,
            title="Betweenness Centrality Distribution",
            labels={'x': 'Betweenness Centrality', 'y': 'Number of Airports'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Summary statistics
    st.write("**📋 Statistical Summary:**")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        st.metric("🎯 Highest Degree Centrality", 
                 f"{max(degree_centrality.values()):.4f}",
                 f"Airport: {max(degree_centrality, key=degree_centrality.get)}")
    
    with summary_col2:
        st.metric("🌉 Highest Betweenness Centrality", 
                 f"{max(betweenness_centrality.values()):.4f}",
                 f"Airport: {max(betweenness_centrality, key=betweenness_centrality.get)}")
    
    with summary_col3:
        st.metric("🎯 Highest Closeness Centrality", 
                 f"{max(closeness_centrality.values()):.4f}",
                 f"Airport: {max(closeness_centrality, key=closeness_centrality.get)}")

def network_structure_analysis(metrics, airport_network):
    """Analyze network structure including clustering and degree distribution"""
    st.subheader("🏗️ Network Structure Analysis")
    
    undirected_graph = airport_network.graph.to_undirected()
    
    # Calculate clustering coefficient
    try:
        global_clustering = nx.average_clustering(undirected_graph)
        transitivity = nx.transitivity(undirected_graph)
    except:
        global_clustering = 0
        transitivity = 0
    
    # Calculate degree distribution
    degrees = [undirected_graph.degree(node) for node in undirected_graph.nodes()]
    degree_counts = {}
    for degree in degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1
    
    # Test for power-law distribution
    degree_sequence = sorted(degrees, reverse=True)
    power_law_fit = analyze_power_law_distribution(degree_sequence)
    
    # Display clustering metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔗 Global Clustering Coefficient", f"{global_clustering:.4f}")
        st.caption("Measures local interconnectedness")
        
    with col2:
        st.metric("🌐 Network Transitivity", f"{transitivity:.4f}")
        st.caption("Global clustering measure")
        
    with col3:
        st.metric("📈 Power-Law Alpha", f"{power_law_fit['alpha']:.2f}" if power_law_fit['alpha'] else "N/A")
        st.caption("Power-law exponent")
    
    # Degree distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Degree Distribution Analysis**")
        
        # Plot degree distribution
        degrees_sorted = sorted(degree_counts.keys())
        counts = [degree_counts[d] for d in degrees_sorted]
        
        fig = px.scatter(
            x=degrees_sorted,
            y=counts,
            title="Degree Distribution",
            labels={'x': 'Degree (Number of Connections)', 'y': 'Number of Airports'},
            log_x=True,
            log_y=True
        )
        fig.update_traces(mode='markers+lines')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Power-law fit quality
        if power_law_fit['r_squared']:
            st.metric("🎯 Power-Law R²", f"{power_law_fit['r_squared']:.3f}")
            st.caption("Goodness of fit (higher = better power-law)")
    
    with col2:
        st.write("**🎯 Network Properties**")
        
        # Small-world properties
        try:
            # Calculate small-world metrics
            if nx.is_connected(undirected_graph):
                # Random graph with same nodes and edges for comparison
                n_nodes = undirected_graph.number_of_nodes()
                n_edges = undirected_graph.number_of_edges()
                random_clustering = (2 * n_edges) / (n_nodes * (n_nodes - 1))
                
                # Small-world coefficient
                small_world_sigma = global_clustering / random_clustering if random_clustering > 0 else 0
                
                st.metric("🌍 Small-World Sigma", f"{small_world_sigma:.2f}")
                st.caption("σ > 1 indicates small-world network")
            else:
                st.info("Small-world analysis requires connected graph")
                
        except Exception as e:
            st.warning("Could not calculate small-world properties")
        
        # Network type classification
        network_type = classify_network_type(global_clustering, power_law_fit, metrics.get('avg_path_length'))
        st.info(f"**Network Type:** {network_type}")
        
        # Degree statistics
        st.write("**📈 Degree Statistics:**")
        degree_stats = {
            'Min Degree': min(degrees),
            'Max Degree': max(degrees),
            'Mean Degree': np.mean(degrees),
            'Median Degree': np.median(degrees),
            'Std Deviation': np.std(degrees)
        }
        
        for stat_name, value in degree_stats.items():
            st.text(f"{stat_name}: {value:.2f}")


def analyze_power_law_distribution(degree_sequence):
    """Analyze if degree distribution follows power law"""
    try:
        # Remove zeros and convert to numpy array
        degrees = np.array([d for d in degree_sequence if d > 0])
        
        if len(degrees) < 10:  # Need sufficient data points
            return {'alpha': None, 'r_squared': None, 'is_power_law': False}
        
        # Fit power law: P(k) ~ k^(-alpha)
        # Taking log: log(P(k)) = -alpha * log(k) + C
        log_degrees = np.log(degrees)
        degree_counts = {}
        for d in degrees:
            degree_counts[d] = degree_counts.get(d, 0) + 1
        
        unique_degrees = sorted(degree_counts.keys())
        log_unique_degrees = np.log(unique_degrees)
        log_probabilities = np.log([degree_counts[d] / len(degrees) for d in unique_degrees])
        
        # Linear regression in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_unique_degrees, log_probabilities)
        alpha = -slope  # Power law exponent
        r_squared = r_value ** 2
        
        # Consider it power-law if R² > 0.8 and alpha > 1
        is_power_law = r_squared > 0.8 and alpha > 1
        
        return {
            'alpha': alpha,
            'r_squared': r_squared,
            'is_power_law': is_power_law,
            'p_value': p_value
        }
        
    except Exception as e:
        return {'alpha': None, 'r_squared': None, 'is_power_law': False}


def classify_network_type(clustering_coeff, power_law_fit, avg_path_length):
    """Classify network type based on structural properties"""
    
    # High clustering coefficient threshold
    high_clustering = clustering_coeff > 0.3
    
    # Power-law degree distribution
    is_scale_free = power_law_fit.get('is_power_law', False)
    
    # Short average path length (log(N) or smaller)
    # Assuming we have this info from metrics
    short_paths = True  # Simplified for now
    
    if is_scale_free and high_clustering:
        return "🌐 Scale-Free Small-World Network"
    elif is_scale_free:
        return "📈 Scale-Free Network"
    elif high_clustering and short_paths:
        return "🔗 Small-World Network"
    elif high_clustering:
        return "🏘️ Highly Clustered Network"
    elif clustering_coeff < 0.1:
        return "🌳 Tree-like Network"
    else:
        return "📊 Random-like Network"
