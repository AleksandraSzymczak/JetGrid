from copy import copy
import polars as pl
import networkx as nx

class AirportNetwork:
    def __init__(self, filter_countries=None):
        self.countries = filter_countries

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
        self.filter_countries(self.countries)


    def filter_countries(self, countries: list):
        if countries:
            self.airports = self.airports.filter(pl.col("Country").is_in(countries))
            print(f"Filtered airports to countries: {countries}. Remaining airports: {len(self.airports)}")
            # Get the set of valid IATA codes from airports dataframe
            valid_iata_codes = set(self.airports['IATA Code'].drop_nans())

            # Filter flight data to include only flights where both departure and arrival codes 
            # are in the airports dataframe
            filtered_flight_data = self.flights.filter(
                (self.flights['Departure'].is_in(valid_iata_codes)) &
                (self.flights['Arrival'].is_in(valid_iata_codes))
            )

            print(f"Original flight data records: {len(self.flights)}")
            print(f"Filtered flight data records: {len(filtered_flight_data)}")
            print(f"Records removed: {len(self.flights) - len(filtered_flight_data)}")
            self.flights = filtered_flight_data
        else:
            print("No country filter applied.")


    def prepare_vertices(self):
        # Get all unique airport codes from flight data (both departure and arrival)
        departure_codes = self.flights.select(pl.col("Departure").alias("iata_code")).drop_nulls()
        arrival_codes = self.flights.select(pl.col("Arrival").alias("iata_code")).drop_nulls()
        
        # Combine and get unique codes that actually appear in flights
        unique_flight_codes = pl.concat([departure_codes, arrival_codes]).unique()
        # Join with airports data to get airport details for these codes
        self.vertices = unique_flight_codes.join(
            self.airports.select([
                pl.col("IATA Code"),
                pl.col("Airport Name").alias("name"),
                pl.col("City").alias("city"),
                pl.col("Country").alias("country"),
                pl.col("Latitude").alias("latitude"),
                pl.col("Longitude").alias("longitude")
            ]),
            left_on="iata_code",
            right_on="IATA Code",
            how="left"
        )

        # Filter out any codes that don't have airport data
        self.vertices = self.vertices.drop_nulls(subset=["name"])


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
        ])
        
        # First, identify bidirectional routes BEFORE deduplication
        original_routes = self.flights.select(["Departure", "Arrival"]).unique()
        routes_set = set(original_routes.iter_rows())
        
        # Function to check if a route is bidirectional
        def is_bidirectional(departure, arrival):
            return (arrival, departure) in routes_set
        
        # Create a normalized route pair identifier
        # Always put the lexicographically smaller airport first
        self.edges = self.edges.with_columns([
            pl.when(pl.col("departure") <= pl.col("arrival"))
            .then(pl.col("departure"))
            .otherwise(pl.col("arrival"))
            .alias("route_start"),
            
            pl.when(pl.col("departure") <= pl.col("arrival"))
            .then(pl.col("arrival"))
            .otherwise(pl.col("departure"))
            .alias("route_end")
        ])
        
        # Remove duplicates based on the normalized route pair
        self.edges = self.edges.unique(subset=["route_start", "route_end"])
        
        # Use normalized departure/arrival
        self.edges = self.edges.with_columns([
            pl.col("route_start").alias("departure"),
            pl.col("route_end").alias("arrival")
        ]).drop(["route_start", "route_end"])
        
        # Add bidirectional attribute
        self.edges = self.edges.with_columns([
            pl.struct(["departure", "arrival"])
            .map_elements(
                lambda x: is_bidirectional(x["departure"], x["arrival"]) or is_bidirectional(x["arrival"], x["departure"]),
                return_dtype=pl.Boolean
            )
            .alias("is_bidirectional")
        ])
        
        # Count bidirectional vs unidirectional routes
        bidirectional_count = len(self.edges.filter(pl.col("is_bidirectional") == True))
        unidirectional_count = len(self.edges.filter(pl.col("is_bidirectional") == False))
        
        print(f"Unique routes after deduplication: {len(self.edges)}")
        print(f"Bidirectional routes: {bidirectional_count}")
        print(f"Unidirectional routes: {unidirectional_count}")
        # # Get all unique route pairs
        # routes_df = self.edges.select(["departure", "arrival"]).unique()
        # routes = set(routes_df.iter_rows())
        
        # # Function to check if a route is bidirectional
        # def is_bidirectional(departure, arrival):
        #     return (arrival, departure) in routes
        
        # # Add color column
        # self.edges = self.edges.with_columns([
        #     pl.struct(["departure", "arrival"])
        #     .map_elements(lambda x: "r" if is_bidirectional(x["departure"], x["arrival"]) else "r")
        #     .alias("color")
        # ])

    def build_graph(self):
        # Filter out nodes without coordinates
        self.vertices = self.vertices.filter(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())

        self.graph = nx.from_pandas_edgelist(
            self.edges.to_pandas(),
            source="departure",
            target="arrival",
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
                "longitude": row[5],
                "iata_code": iata_code
            }
            if iata_code in self.graph:
                self.graph.nodes[iata_code].update(attrs)
            else:
                self.graph.add_node(iata_code, **attrs)

    def __copy__(self):
        """Shallow copy of the AirportNetwork"""
        new_network = AirportNetwork(filter_countries=self.countries)
        
        # Copy all attributes if they exist
        if hasattr(self, 'airports'):
            new_network.airports = self.airports.clone()
        if hasattr(self, 'flights'):
            new_network.flights = self.flights.clone()
        if hasattr(self, 'vertices'):
            new_network.vertices = self.vertices.clone()
        if hasattr(self, 'edges'):
            new_network.edges = self.edges.clone()
        if hasattr(self, 'graph'):
            new_network.graph = self.graph.copy()
        
        return new_network

    def __deepcopy__(self, memo):
        """Deep copy of the AirportNetwork"""
        new_network = AirportNetwork(filter_countries=copy.deepcopy(self.countries, memo))
        
        # Deep copy all attributes if they exist
        if hasattr(self, 'airports'):
            new_network.airports = self.airports.clone()
        if hasattr(self, 'flights'):
            new_network.flights = self.flights.clone()
        if hasattr(self, 'vertices'):
            new_network.vertices = self.vertices.clone()
        if hasattr(self, 'edges'):
            new_network.edges = self.edges.clone()
        if hasattr(self, 'graph'):
            new_network.graph = copy.deepcopy(self.graph, memo)
        
        return new_network

    def copy(self):
        """Public method to create a copy of the network"""
        return copy.copy(self)

    def deepcopy(self):
        """Public method to create a deep copy of the network"""
        return copy.deepcopy(self)

    @classmethod
    def create_from_existing(cls, original_network, modified_graph=None):
        """Class method to create a new network from an existing one"""
        new_network = cls(filter_countries=original_network.countries)
        
        # Copy data attributes
        if hasattr(original_network, 'airports'):
            new_network.airports = original_network.airports.clone()
        if hasattr(original_network, 'flights'):
            new_network.flights = original_network.flights.clone()
        if hasattr(original_network, 'vertices'):
            new_network.vertices = original_network.vertices.clone()
        if hasattr(original_network, 'edges'):
            new_network.edges = original_network.edges.clone()
        
        # Use provided graph or copy original
        if modified_graph is not None:
            new_network.graph = modified_graph.copy()
        elif hasattr(original_network, 'graph'):
            new_network.graph = original_network.graph.copy()
        
        return new_network