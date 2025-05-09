from Fetch_Calculate import fetch_raw_data
from Excel_Sheets import ExcelWriter
from Predicted_Data import PrecipitationPredictorAI

import ee
import queue
import threading
import pandas as pd
import xlsxwriter
import os
import webbrowser

def main():
    # Initialize Google Earth Engine
    ee.Authenticate()
    ee.Initialize(project='qattara-depression2')

    # Create a queue for sharing data between threads
    data_queue = queue.Queue()

    # Start threads
    producer_thread = threading.Thread(target=fetch_raw_data, args=(data_queue,))# Exchange the Q data between two threads
    consumer_thread = threading.Thread(target=write_to_excel_and_plot, args=(data_queue,))

    producer_thread.start()
    consumer_thread.start()

    producer_thread.join()
    consumer_thread.join()

# Consumer thread logic
def write_to_excel_and_plot(data_queue):
    try:
        workbook = xlsxwriter.Workbook('Seasonal_Data_2019_2024.xlsx')

        # Collect all data from the queue
        detailed_data_list = []
        while True:
            detailed_data = data_queue.get()
            if detailed_data is None:
                break
            if detailed_data:  # Ensure detailed_data is not None or empty
                detailed_data_list.extend(detailed_data)
            else:
                print("Received empty data.")
                continue

        # Check if there is any valid data collected
        if detailed_data_list:
            detailed_df = pd.DataFrame(detailed_data_list)
        else:
            print("No valid data to write to Excel.")
            return

        # Group the data by season for the seasonal data sheets
        seasons_data = {}
        for _, row in detailed_df.iterrows():
            season = row['Season']
            if season not in seasons_data:
                seasons_data[season] = []
            seasons_data[season].append(row)

        # workbook isn't subscribed (مينفعش ابعته لكذا فانكشن)
        excel_writer = ExcelWriter(workbook, detailed_df, seasons_data)

        # Write seasonal data to separate sheets
        excel_writer.write_seasonal_data()

        # ------
        # Aggregating data for each season (mean for temperature, humidity, and evaporation; sum for precipitation)
        aggregated_season_data=excel_writer.write_aggregated_season_data()

        # ------
        # Aggregating data for all seasons (mean for temperature, humidity, and evaporation; sum for precipitation)
        aggregated_all_seasons = excel_writer.write_aggregated_all_seasons_data(aggregated_season_data)

        # ------
        # Predicted data sheets for each season
        # predicted_df = excel_writer.write_predicted_data()

        # Initialize the PrecipitationPredictorAI class
        predictor = PrecipitationPredictorAI(workbook, detailed_df, seasons_data)

        # Call the write_predicted_data function to generate predictions and write to Excel
        predicted_df = predictor.write_predicted_data()

        # ------
        #Aggregated predicted data for each year
        excel_writer.write_aggregated_predicted_data(predicted_df,aggregated_all_seasons)
        # ======
        excel_writer.close()
        file_path = os.path.abspath('Seasonal_Data_2019_2024.xlsx')
        webbrowser.open(f'file://{file_path}')

    except Exception as e:
        print("")
        print(f"Error in writing to Excel: {e}")

if __name__ == "__main__":
    main()
