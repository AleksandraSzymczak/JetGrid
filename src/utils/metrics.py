import streamlit as st
import networkx as nx
import polars as pl


def centrality_comparison_experiment(airport_network):
    """Experiment: Compare different centrality measures"""
    st.subheader("📊 Centrality Comparison")
    st.write("**Question:** Which airports are most important by different measures?")
    
    if st.button("🔄 Calculate All Centralities"):
        with st.spinner("Calculating centralities..."):
            # Calculate different centralities
            degree_cent = nx.degree_centrality(airport_network.graph)
            betweenness_cent = nx.betweenness_centrality(airport_network.graph, k=min(1000, airport_network.graph.number_of_nodes()))
            closeness_cent = nx.closeness_centrality(airport_network.graph)

            # Create comparison dataframe
            airports = list(airport_network.graph.nodes())
            comparison_data = []
            
            for airport in airports:
                airport_data = airport_network.graph.nodes.get(airport, {})
                comparison_data.append({
                    'Airport': airport,
                    'Name': airport_data.get('name', 'Unknown'),
                    'Country': airport_data.get('country', 'Unknown'),
                    'Degree': degree_cent.get(airport, 0),
                    'Betweenness': betweenness_cent.get(airport, 0),
                    'Closeness': closeness_cent.get(airport, 0)
                })
            
            df = pl.DataFrame(comparison_data)
            
            # Show top airports by each measure
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Top by Degree**")
                top_degree = df.nlargest(10, 'Degree')[['Airport', 'Country', 'Degree']]
                st.dataframe(top_degree, use_container_width=True)
            
            with col2:
                st.write("**Top by Betweenness**")
                top_between = df.nlargest(10, 'Betweenness')[['Airport', 'Country', 'Betweenness']]
                st.dataframe(top_between, use_container_width=True)
            
            with col3:
                st.write("**Top by Closeness**")
                top_close = df.nlargest(10, 'Closeness')[['Airport', 'Country', 'Closeness']]
                st.dataframe(top_close, use_container_width=True)
            
            # Correlation analysis
            correlation_matrix = df[['Degree', 'Betweenness', 'Closeness']].corr()
            
            fig = px.imshow(correlation_matrix, 
                        title="Centrality Measures Correlation",
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
            st.plotly_chart(fig, use_container_width=True)