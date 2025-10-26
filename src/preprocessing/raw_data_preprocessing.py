import csv
from turtle import pd
from typing import List, Set
import polars as pl
import yaml

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

def preprocess__data(csv_flight_path: str, output_flight_path: str, csv_airport_path: str, output_airport_path: str):
    df_flights = pl.read_csv(csv_flight_path, null_values=["\\N", ""])
    df_airports = pl.read_csv(csv_airport_path, null_values=["\\N", ""])
    df_filtered_flights = filter_invalid_flights(df_flights)
    df_filtered_flights = filter_invalid_flights_airports(df_filtered_flights, df_airports)
    create_countries_file_from_airports(df_airports)
    df_filtered_airports = filter_airports_without_iata(df_airports)
    df_filtered_flights.write_csv(output_flight_path)
    df_filtered_airports.write_csv(output_airport_path)

# IATA code empty - primarily assigned to airports that handle commercial passenger traffic
# also filter out rows with no arrival or departure code
def filter_invalid_flights(df: pl.DataFrame) -> pl.DataFrame:
    """
    Remove flights with missing arrival or departure codes/names
    """
    # Filter out rows with null Departure Code, Arrival Code, Departure, or Arrival
    df_filtered = df.filter(
        pl.col("Departure Code").is_not_null() & 
        pl.col("Arrival Code").is_not_null() &
        pl.col("Departure").is_not_null() & 
        pl.col("Arrival").is_not_null()
    )
    removed_count = df.height - df_filtered.height
    if removed_count > 0:
        print(f"Removed {removed_count} invalid flight records with missing codes or airport names")
    else:
        print("No invalid flight records found")
    
    return df_filtered

def filter_invalid_flights_airports(df_flights: pl.DataFrame, df_airports: pl.DataFrame) -> pl.DataFrame:
    """
    Filter flights to only include those between airports that exist in the airports dataset
    """
    # Get the set of valid IATA codes from airports.csv
    valid_iata_codes = df_airports.filter(pl.col("IATA Code").is_not_null())["IATA Code"].to_list()
    
    print(f"Found {len(valid_iata_codes)} valid airports with IATA codes")
    
    # Filter flight_data to only include rows where both departure and arrival airports exist
    # Use Polars .filter() method instead of bracket notation
    filtered_flight_data = df_flights.filter(
        pl.col("Departure").is_in(valid_iata_codes) &
        pl.col("Arrival").is_in(valid_iata_codes)
    )

    print(f"Original flight data rows: {df_flights.height}")
    print(f"Filtered flight data rows: {filtered_flight_data.height}")
    print(f"Removed {df_flights.height - filtered_flight_data.height} rows")

    
    return filtered_flight_data


def create_countries_file_from_airports(df_airports: pl.DataFrame):
    unique_countries = df_airports.filter(
        pl.col("Country").is_not_null()
    )["Country"].unique().sort().to_list()

    countries_data = {
        "countries": unique_countries,
        "total_count": len(unique_countries),
        "metadata": {
            "source": "airports_dataset",
            "description": "Unique countries from global airports data",
        }
    }
    
    # Save to YAML file
    with open("data/countries.yaml", 'w', encoding='utf-8') as file:
        yaml.dump(countries_data, file, default_flow_style=False, allow_unicode=True)

    print(f"Saved {len(unique_countries)} unique countries to data/countries.yaml")


def filter_airports_without_iata(df_airports: pl.DataFrame) -> pl.DataFrame:
    """
    Remove airports that don't have IATA codes (commercial airports only)
    """
    # Filter out rows with null or empty IATA codes
    df_filtered = df_airports.filter(
        pl.col("IATA Code").is_not_null() &
        (pl.col("IATA Code") != "") &
        (pl.col("IATA Code") != "\\N") &  # Add this line to filter out \N strings
        (pl.col("IATA Code") != "N") &    # Sometimes it might be just N without backslash
        ~pl.col("IATA Code").str.contains(r"^\d+$")  # Also remove numeric-only codes
    )
    removed_count = df_airports.height - df_filtered.height
    if removed_count > 0:
        print(f"Removed {removed_count} airports without valid IATA codes")
        print(f"Remaining airports: {df_filtered.height}")
    else:
        print("All airports already have valid IATA codes")
    return df_filtered

