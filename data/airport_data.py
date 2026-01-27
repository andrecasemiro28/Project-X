"""
Airport Data Module
===================
This module handles loading airport data from various sources.
Currently loads from a local JSON file, but can be easily extended
to load from an API or database.

To replace with real data:
1. Update the AIRPORTS_SOURCE variable to 'api' or 'database'
2. Implement the corresponding load function
3. Ensure the data follows the same schema:
   - city_name: str
   - airport_code: str (IATA code)
   - airport_name: str
"""

import json
import os
from typing import List, Dict

# Configuration: Change this to switch data sources
# Options: 'json', 'api', 'database'
AIRPORTS_SOURCE = 'json'

# Path to the JSON file (relative to project root)
AIRPORTS_JSON_PATH = os.path.join(os.path.dirname(__file__), 'airports.json')


def load_airports_from_json() -> List[Dict]:
    """Load airports from local JSON file."""
    try:
        with open(AIRPORTS_JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []
    except json.JSONDecodeError:
        return []


def load_airports_from_api() -> List[Dict]:
    """
    Load airports from an external API.
    TODO: Implement when API is available.
    """
    # Example implementation:
    # import requests
    # response = requests.get('https://api.example.com/airports')
    # return response.json()
    raise NotImplementedError("API source not yet implemented")


def load_airports_from_database() -> List[Dict]:
    """
    Load airports from a database.
    TODO: Implement when database is available.
    """
    raise NotImplementedError("Database source not yet implemented")


def get_airports() -> List[Dict]:
    """
    Main function to get airport data.
    Automatically selects the appropriate data source.
    """
    if AIRPORTS_SOURCE == 'json':
        return load_airports_from_json()
    elif AIRPORTS_SOURCE == 'api':
        return load_airports_from_api()
    elif AIRPORTS_SOURCE == 'database':
        return load_airports_from_database()
    else:
        return load_airports_from_json()


def get_airport_options() -> List[str]:
    """
    Get formatted airport options for dropdown selection.
    Format: "City Name - CODE (Airport Name)"
    """
    airports = get_airports()
    options = []
    for airport in airports:
        option = f"{airport['city_name']} - {airport['airport_code']} ({airport['airport_name']})"
        options.append(option)
    return sorted(options)


def parse_airport_selection(selection: str) -> Dict:
    """
    Parse a dropdown selection back to airport data.
    Returns dict with city_name, airport_code, airport_name.
    """
    if not selection or ' - ' not in selection:
        return {'city_name': '', 'airport_code': '', 'airport_name': ''}

    # Format: "City Name - CODE (Airport Name)"
    parts = selection.split(' - ')
    city_name = parts[0]
    code_and_name = parts[1]

    # Extract code and name
    airport_code = code_and_name.split(' (')[0]
    airport_name = code_and_name.split(' (')[1].rstrip(')')

    return {
        'city_name': city_name,
        'airport_code': airport_code,
        'airport_name': airport_name
    }


def get_route_code(origin_selection: str, destination_selection: str) -> str:
    """
    Generate route code from origin and destination selections.
    Format: "XXX-YYY" where XXX and YYY are airport codes.
    """
    origin = parse_airport_selection(origin_selection)
    destination = parse_airport_selection(destination_selection)
    return f"{origin['airport_code']}-{destination['airport_code']}"
