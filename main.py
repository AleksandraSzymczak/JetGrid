import networkx as nx
import polars as pl
import csv
from typing import List
import matplotlib.pyplot as plt
import streamlit as st
import folium



class AirportNetwork:
    def load_data(self, airports_csv_path, flights_csv_path):
        self.airports = pl.read_csv(
            airports_csv_path,
            null_values=["\\N", ""],
            ignore_errors=True
        )
        self.flights = pl.read_csv(
            flights_csv_path,
            null_values=["\\N", ""],
            ignore_errors=True
        )
    
    def prepare_vertices(self):
        # Prepare vertices (airports)
        self.vertices = self.airports.select([
            pl.col("IATA Code").alias("iata_code"),
            pl.col("Airport Name").alias("name"),
            pl.col("City").alias("city"),
            pl.col("Country").alias("country"),
            pl.col("Latitude").alias("latitude"),
            pl.col("Longitude").alias("longitude")
        ]).drop_nulls(subset=["iata_code"]).unique(subset=["iata_code"])

    def prepare_edges(self):
        # Prepare edges (flights)
        self.edges = self.flights.select([
            pl.col("Flight Code").alias("flight_code"),
            pl.col("Departure").alias("departure"),
            pl.col("Departure Code").alias("departure_code"),
            pl.col("Arrival").alias("arrival"),
            pl.col("Arrival Code").alias("arrival_code"),
            pl.col("Status").alias("status"),
            pl.col("Flight Type").alias("flight_type")
        ]).drop_nulls(subset=["flight_code"]).unique(subset=["flight_code"])

    def build_graph(self):
        # Filter out nodes without coordinates
        self.vertices = self.vertices.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
        self.graph = nx.from_pandas_edgelist(
            self.edges.to_pandas(),
            source="departure_code",
            target="arrival_code",
            edge_attr=True,
            create_using=nx.DiGraph()
        )
        for row in self.vertices.iter_rows():
            iata_code = row[0]
            attrs = {
                "name": row[1],
                "city": row[2],
                "country": row[3],
                "latitude": row[4],
                "longitude": row[5]
            }
            if iata_code in self.graph:
                self.graph.nodes[iata_code].update(attrs)
            else:
                self.graph.add_node(iata_code, **attrs)

def load_raw_data_to_csv(raw_data_path: str, output_data_path: str, headers: List):
    # read raw data
    with open(raw_data_path, "r") as f:
        readers = csv.reader(f)
        data = [row for row in readers]
    # save data to csv
    with open(output_data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)


def main():
    g = nx.Graph()


if __name__ == "__main__":
    headers = [
        'Flight Code',
        'Flight Number',
        'Departure',
        'Departure Code',
        'Arrival',
        'Arrival Code',
        'Unknown',
        'Status',
        'Flight Type'
    ]
    load_raw_data_to_csv(
        raw_data_path="raw_data/flight_data.txt",
        output_data_path="data/flight_data.csv",
        headers=headers
    )
    headers = [
        "ID",
        "Airport Name",
        "City",
        "Country",
        "IATA Code",
        "ICAO Code",
        "Latitude",
        "Longitude",
        "Altitude (ft)",
        "Timezone Offset",
        "Location Type",
        "Timezone",
        "Airport Type",
        "Source"
    ]
    load_raw_data_to_csv(
        raw_data_path="raw_data/airports.txt",
        output_data_path="data/airports.csv",
        headers=headers
    )
    airport_network = AirportNetwork()
    airport_network.load_data(
        airports_csv_path="data/airports.csv",
        flights_csv_path="data/flight_data.csv"
    )
    print(airport_network.airports.head())
    print(airport_network.flights.head())

    airport_network.prepare_vertices()
    airport_network.prepare_edges()
    print(airport_network.vertices.head())
    print(airport_network.edges.head())
    airport_network.build_graph()

    # Create position dictionary using longitude and latitude
    pos = {}
    nodes_without_coords = []
    
    for node in airport_network.graph.nodes():
        if 'longitude' in airport_network.graph.nodes[node] and 'latitude' in airport_network.graph.nodes[node]:
            lon = airport_network.graph.nodes[node]['longitude']
            lat = airport_network.graph.nodes[node]['latitude']
            if lon is not None and lat is not None:
                pos[node] = (lon, lat)
            else:
                nodes_without_coords.append(node)
        else:
            nodes_without_coords.append(node)
    
    # Remove nodes without coordinates from the graph
    print(f"Removing {len(nodes_without_coords)} nodes without coordinates")
    airport_network.graph.remove_nodes_from(nodes_without_coords)
    

    plt.figure(figsize=(15, 10))
    nx.draw(
        airport_network.graph, 
        pos=pos,
        with_labels=False,  # Turn off labels for better visibility
        node_size=30,
        node_color='red',
        edge_color='blue',
        alpha=0.6,
    )
    plt.show()
    st.title("Airport Network Visualization")
    st.write("Sample Airports Data:")
st.dataframe(airport_network.vertices.head(10).to_pandas())

st.write("Sample Flights Data:")
st.dataframe(airport_network.edges.head(10).to_pandas())

st.write("Graph Info:")
st.text(f"Number of nodes: {airport_network.graph.number_of_nodes()}")
st.text(f"Number of edges: {airport_network.graph.number_of_edges()}")

st.pyplot(plt)
# Add dots for each airport
# Create base map (centered on the world)
m = folium.Map(location=[20, 0], zoom_start=2)
for node in airport_network.graph.nodes():
    folium.CircleMarker(
        location=[airport_network.graph.nodes[node]["latitude"], airport_network.graph.nodes[node]["longitude"]],
        radius=3,
        popup=node,
        color='blue',
        fill=True,
        fill_color='blue'
    ).add_to(m)

import streamlit as st

# ...existing code...

# Read HTML from a file
with open("airports_map.html", "r") as f:
    html_content = f.read()

# Display HTML in Streamlit
st.components.v1.html(html_content, height=600, scrolling=True)
