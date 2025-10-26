import networkx as nx
from typing import List
import matplotlib.pyplot as plt
import streamlit as st
import folium
from config import FLIGHT_HEADERS, AIRPORT_HEADERS, RAW_DATA_PATHS, OUTPUT_DATA_PATHS
from src.preprocessing.raw_data_preprocessing import (
    load_raw_data_to_csv, 
    preprocess__data,
)
from src.AirportNetwork import AirportNetwork
import yaml
from src.components.ui_metrics import (
    generate_countries_list,
    metrics_display
)
from src.components.map_generator import generate_map
from src.tabs.experiments import experiment_tab_flow

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess data with caching to avoid reloading"""
    load_raw_data_to_csv(
        raw_data_path=RAW_DATA_PATHS['flights'],
        output_data_path=OUTPUT_DATA_PATHS['flights'],
        headers=FLIGHT_HEADERS
    )
    
    load_raw_data_to_csv(
        raw_data_path=RAW_DATA_PATHS['airports'],
        output_data_path=OUTPUT_DATA_PATHS['airports'],
        headers=AIRPORT_HEADERS
    )
    
    preprocess__data(
        csv_flight_path=OUTPUT_DATA_PATHS['flights'],
        output_flight_path=OUTPUT_DATA_PATHS['flights'],
        csv_airport_path=OUTPUT_DATA_PATHS['airports'],
        output_airport_path=OUTPUT_DATA_PATHS['airports']
    )
    return True

@st.cache_resource
def build_airport_network(selected_countries):
    """Build and cache airport network to prevent rebuilds"""
    airport_network = AirportNetwork(filter_countries=selected_countries)
    airport_network.load_data(
        airports_csv_path=OUTPUT_DATA_PATHS["airports"],
        flights_csv_path=OUTPUT_DATA_PATHS["flights"]
    )
    airport_network.prepare_vertices()
    airport_network.prepare_edges()
    airport_network.build_graph()
    return airport_network

if __name__ == "__main__":
    # Set page config FIRST
    st.set_page_config(
        page_title="Airport Network Analysis", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load and preprocess data ONCE at startup
    data_loaded = load_and_preprocess_data()
    
    # Sidebar navigation
    st.sidebar.title("🛫 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🗺️ Network Map", "📊 Analytics Dashboard", "🧪 Experiments", "ℹ️ More info"],
        key="main_page_selector"
    )
    
    if data_loaded:
        # Generate country selection list    
        selected_countries = generate_countries_list()
        
        # Convert to tuple for caching (lists aren't hashable)
        countries_tuple = tuple(selected_countries) if selected_countries else None
        
        if selected_countries:
            st.info(f"🔍 Building network for: {', '.join(selected_countries)}")
        else:
            st.info("🌐 Building global network (all countries)")

        # Build network only once per country selection (cached)
        with st.spinner("Building airport network..."):
            airport_network = build_airport_network(countries_tuple)

        # Generate map only once per network (cached)  
        if 'map_generated' not in st.session_state or st.session_state.get('last_countries') != countries_tuple:
            with st.spinner("Generating map..."):
                generate_map(airport_network)
                st.session_state.map_generated = True
                st.session_state.last_countries = countries_tuple

        # Read HTML map
        try:
            with open("airports_map.html", "r") as f:
                html_content = f.read()
        except FileNotFoundError:
            html_content = "<p>Map not available</p>"

        # Display content based on selected page
        if page == "🗺️ Network Map":
            st.title("🗺️ Airport Network Visualization")
            st.components.v1.html(html_content, height=600, scrolling=True)

        elif page == "🧪 Experiments":
            st.title("🧪 Network Experiments")
            experiment_tab_flow(airport_network)

        elif page == "📊 Analytics Dashboard":
            st.title("📊 Analytics Dashboard")
            metrics_display(airport_network)

        elif page == "ℹ️ More info":
            st.title("ℹ️ About This Application")
            st.markdown("""

            This application visualizes and analyzes a global network of airports and flights using Streamlit and NetworkX.

            **Data Sources:**
            - Flight and Airport data sourced from https://openflights.org/data

            Data is not up-to-date and is used for demonstration purposes only - this is state from 2009,
                        only commercial flights with IATA codes are included.
            """)
            st.write("**Sample Airports Data:**")
            st.dataframe(airport_network.vertices.head(10).to_pandas())
            st.write("**Sample Flights Data:**")
            st.dataframe(airport_network.edges.head(10).to_pandas())
    else:
        st.error("❌ Failed to load data. Please check your data files.")