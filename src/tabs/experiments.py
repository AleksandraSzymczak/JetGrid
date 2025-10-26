from src.experiments.interactive_experiments import Experiment
from src.components.map_generator import generate_map
import streamlit as st

def experiment_tab_flow(airport_network):
    """Function to handle the experiment tab flow"""
    # Display network experiments
    Experiment(airport_network).display_network_experiments()