#                         Fetch climate data then fetch raw data and calculate the relative & specific humidity
import queue
import numpy as np
import ee
import pandas as pd
import calendar
from datetime import datetime

# Queue for data processing
data_queue = queue.Queue()

# Functions for humidity calculations
def calculate_relative_humidity(temp_k, dewpoint_k):
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    actual_vapor_pressure = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    saturation_vapor_pressure = 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
    return np.clip((actual_vapor_pressure / saturation_vapor_pressure) * 100, 0, 100)  # Return percentage


def calculate_specific_humidity(temp_k, dewpoint_k, pressure):
    pressure_hpa = pressure / 100  # Convert Pa to hPa
    dewpoint_c = dewpoint_k - 273.15
    actual_vapor_pressure = 6.112 * np.exp((17.67 * dewpoint_c) / (dewpoint_c + 243.5))
    q = (0.622 * actual_vapor_pressure) / (pressure_hpa - (0.378 * actual_vapor_pressure))
    return q * 1000  # Convert to g/kg

# -------------------------------------------------
# Climate data fetching function
def fetch_climate_data(latitude, longitude, start_date, end_date):
    try:
        climateVariables = [
            'temperature_2m',
            'temperature_2m_min',
            'temperature_2m_max',
            'dewpoint_temperature_2m',
            'surface_pressure',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m',
            'total_precipitation_sum',
            'soil_temperature_level_1',
            'total_evaporation_sum',
            'evaporation_from_bare_soil_sum',
            'evaporation_from_vegetation_transpiration_sum'
        ]
        point = ee.Geometry.Point([longitude, latitude])
        #Using one dataset increases runtime speed instead of using several datasets
        dataset = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
            .select(climateVariables) \
            .filterDate(start_date, end_date) \
            .getRegion(point, 1000) \
            .getInfo()

        if dataset:
            headers = dataset[0]
            records = dataset[1:]
            df = pd.DataFrame(records, columns=headers)
            return df
        else:
            print(f"No data available for {start_date} to {end_date}.")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# -------------------------------------------------
# Fetch seasonal data function

def fetch_raw_data(data_queue):
    latitude = 29.0
    longitude = 31.0

    detailed_data = []
    seasons = {
        'Spring': ('03-01', '05-31'),
        'Summer': ('06-01', '08-31'),
        'Autumn': ('09-01', '11-30'),
        'Winter': ('12-01', '02-29'),
    }

    for year in range(2019, 2025):
        for season, (start_suffix, end_suffix) in seasons.items():
            if season == 'Winter' and not calendar.isleap(year + 1):
                end_suffix = '02-28'

            start_date = f"{year}-{start_suffix}"
            end_date = f"{year + 1 if season == 'Winter' else year}-{end_suffix}"

            print(f"Fetching data for {season} {year} ({start_date} to {end_date})")

            df = fetch_climate_data(latitude, longitude, start_date, end_date)

            if df is None or df.empty:
                print(f"No data for {season} {year}")
                continue

            for _, row in df.iterrows():
                temp_k = row.get('temperature_2m', None)
                dewpoint_k = row.get('dewpoint_temperature_2m', None)
                pressure = row.get('surface_pressure', None)

                if temp_k is not None and dewpoint_k is not None and pressure is not None:
                    relative_humidity = calculate_relative_humidity(temp_k, dewpoint_k)
                    specific_humidity = calculate_specific_humidity(temp_k, dewpoint_k, pressure)
                else:
                    relative_humidity = None
                    specific_humidity = None

                    # Convert the date to the format dd:mm:yy
                date_str = row['time']
                formatted_date = datetime.utcfromtimestamp(date_str / 1000).strftime('%d/%m')

                detailed_data.append({
                    'Season': season,
                    'Year': year,
                    'Date': formatted_date, # row['time']
                    'Temperature': temp_k - 273.15 if temp_k else None,  # Convert to Celsius
                    'Min Temperature': row.get('temperature_2m_min', None) - 273.15 if row.get('temperature_2m_min', None) else None,
                    'Max Temperature': row.get('temperature_2m_max', None) - 273.15 if row.get('temperature_2m_max', None) else None,
                    'Dewpoint': dewpoint_k - 273.15 if dewpoint_k else None,  # Convert to Celsius
                    'Surface Pressure': pressure,
                    'U Wind': row.get('u_component_of_wind_10m', None),
                    'V Wind': row.get('v_component_of_wind_10m', None),
                    'Precipitation': row.get('total_precipitation_sum', None),
                    'Soil Temperature': row.get('soil_temperature_level_1', None) - 273.15 if row.get('soil_temperature_level_1', None) else None,
                    'Evaporation': row.get('total_evaporation_sum', None),
                    'Bare Soil Evaporation': row.get('evaporation_from_bare_soil_sum', None),
                    'Vegetation Evaporation': row.get('evaporation_from_vegetation_transpiration_sum', None),
                    'Relative Humidity (%)': relative_humidity,
                    'Specific Humidity (g/kg)': specific_humidity
                })
        print("")# To separate between years

    if detailed_data:  # Only put in queue if there's data
        data_queue.put(detailed_data)
    else:
        print("No detailed data to put in the queue.")
    data_queue.put(None)