# Configuration file for JetGrid project

# Flight data column headers
FLIGHT_HEADERS = [
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

# Airport data column headers
AIRPORT_HEADERS = [
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

# File paths
RAW_DATA_PATHS = {
    'flights': 'raw_data/flight_data.txt',
    'airports': 'raw_data/airports.txt'
}

OUTPUT_DATA_PATHS = {
    'flights': 'data/flight_data.csv',
    'airports': 'data/airports.csv'
}