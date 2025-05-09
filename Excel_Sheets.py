#                                     WRITE ALL EXCEL SHEETS + PASS DATA TO PLOT FUNCTION
import pandas as pd

from Plotting import plot_weather_data, plot_predicted_weather_data, plot_combined_weather_data

class ExcelWriter:
    def __init__(self, workbook, detailed_df,seasons_data):
        self.workbook = workbook
        self.detailed_df = detailed_df
        self.seasons_data =seasons_data

        # Define color formats for each year
        self.year_colors = {
            2019: self.workbook.add_format({'bg_color': '#FFCCCB'}),
            2020: self.workbook.add_format({'bg_color': '#ADD8E6'}),
            2021: self.workbook.add_format({'bg_color': '#90EE90'}),
            2022: self.workbook.add_format({'bg_color': '#FFFFE0'}),
            2023: self.workbook.add_format({'bg_color': '#F0E68C'}),
            2024: self.workbook.add_format({'bg_color': '#D8BFD8'}),
        }

    # Ensure sheet names are unique before adding them
    # Excel treats sheet names as case-insensitive, so even if there’s a difference in letter case, it will throw an error.
    def ensure_unique_sheet_name(self, base_name):
        # Check if sheet with base_name exists; if it does, append a number to make it unique
        name = base_name
        counter = 1
        while name in self.workbook.sheetnames:
            name = f"{base_name}_{counter}"
            counter += 1
        return name

    # ==============================================
    # Write seasonal data to separate sheets
    def write_seasonal_data(self):
        for season, season_data in self.seasons_data.items():
            sheet_name = self.ensure_unique_sheet_name(season)
            sheet = self.workbook.add_worksheet(sheet_name)

            # Write headers
            headers = list(season_data[0].keys())
            for col, header in enumerate(headers):
                sheet.write(0, col, header)

            # Write rows with year-based color formatting
            for row_idx, row in enumerate(season_data):
                year = row['Year']
                color_format = self.year_colors.get(year, self.workbook.add_format()) # Default format if year is not in year_colors
                for col_idx, header in enumerate(headers):
                    sheet.write(row_idx + 1, col_idx, row[header], color_format)

        print("Seasonal raw data have been written successfully ✅")

    # -----------------------------------------------
    # Aggregated data for each season (mean for temperature, humidity, and evaporation; sum for precipitation)
    def write_aggregated_season_data(self):
        aggregated_season_data = pd.DataFrame()

        for season, season_data in self.seasons_data.items():
            season_df = pd.DataFrame(season_data)
            season_agg = season_df.groupby('Year').agg({
                'Temperature': 'mean',
                'Relative Humidity (%)': 'mean',
                'Evaporation': 'mean',
                'Precipitation': 'sum'
            }).reset_index()

            sheet_name = self.ensure_unique_sheet_name(f"{season}_Aggregated")
            sheet = self.workbook.add_worksheet(sheet_name)
            headers = ["Year", "Temperature", "Relative Humidity", "Evaporation", "Total Precipitation"]
            for col, header in enumerate(headers):
                sheet.write(0, col, header)

            # Write aggregated data with year-based color formatting
            for row_idx, row in season_agg.iterrows():
                year = row['Year']
                color_format = self.year_colors.get(year, self.workbook.add_format()) # Default format if year is not in year_colors

                for col_idx, value in enumerate(row):
                    sheet.write(row_idx + 1, col_idx, value, color_format)

            # Append the aggregated data to the final combined DataFrame
            aggregated_season_data = pd.concat([aggregated_season_data, season_agg])

        print("Aggregated data for each season have been written successfully ✅")
        return aggregated_season_data

    # -----------------------------------------------
    # Aggregating data for all seasons (mean for temperature, humidity, and evaporation; sum for precipitation)
    def write_aggregated_all_seasons_data(self,aggregated_season_data):
        # aggregated_season_data = self.write_aggregated_season_data() will write the code "write_aggregated_season_data" is repeated twice
        aggregated_all_seasons = aggregated_season_data.groupby('Year').agg({
            'Temperature': 'mean',
            'Relative Humidity (%)': 'mean',
            'Evaporation': 'mean',
            'Precipitation': 'sum'
        }).reset_index()

        # Write Aggregated Data for all seasons to a sheet
        sheet = self.workbook.add_worksheet('Aggregated All Seasons')
        headers = ["Year", "Temperature", "Relative Humidity", "Evaporation", "Total Precipitation"]
        for col, header in enumerate(headers):
            sheet.write(0, col, header)

        # Write aggregated data with color formatting
        for row_idx, row in aggregated_all_seasons.iterrows():
            year = row['Year']
            color_format = self.year_colors.get(year, self.workbook.add_format()) # Default format if year is not in year_colors

            for col_idx, value in enumerate(row):
                sheet.write(row_idx + 1, col_idx, value, color_format)

        print("Aggregated data for all seasons have been written successfully ✅")
        plot_weather_data(aggregated_all_seasons, self.workbook)
        return aggregated_all_seasons

    # -----------------------------------------------
    # from sklearn.linear_model import LinearRegression #pip install scikit-learn (Allows you to fit a linear model to the data and obtain the coefficients)
    # basic machine learning model, not typically referred to as an "AI" model in the more advanced sense

    # def fit_regression_model(self):
    #     # Filter rows where the necessary data is available
    #     filtered_df = self.detailed_df.dropna(subset=['Temperature', 'Relative Humidity (%)', 'Evaporation', 'Precipitation'])
    #
    #     # Define the features (X) and target (y)
    #     X = filtered_df[['Temperature', 'Relative Humidity (%)', 'Evaporation']]
    #     y = filtered_df['Precipitation']
    #
    #     # Initialize and fit the regression model
    #     model = LinearRegression()
    #     model.fit(X, y)
    #
    #     # Get the coefficients and intercept
    #     a1, a2, a3 = model.coef_
    #     intercept = model.intercept_
    #
    #     return a1, a2, a3, intercept
    #
    # # Predicted data sheets for each season
    # def write_predicted_data(self):
    #
    #     # Define different color formats for temperature, evaporation, and humidity transitions
    #     temp_color_format = self.workbook.add_format({'bg_color': '#FFC7CE'})  # Light red fill for temperature changes
    #     evap_color_format = self.workbook.add_format({'bg_color': '#C6EFCE'})  # Light green fill for evaporation changes
    #     humidity_color_format = self.workbook.add_format({'bg_color': '#FFEB9C'})  # Light yellow fill for humidity changes
    #     step_size = 0.000001  # step size to control the increment size for evaporation
    #
    #     # Initialize an empty list to store the predicted data
    #     predicted_data = []
    #     a1, a2, a3, intercept = self.fit_regression_model()
    #
    #     # Iterate through each season
    #     for season, season_data in self.seasons_data.items():
    #         # Convert the season_data (which is a list) into a DataFrame
    #         season_df = pd.DataFrame(season_data)
    #
    #         # Ensure the DataFrame has a 'Year' column to group by
    #         if 'Year' not in season_df.columns:
    #             raise ValueError("The DataFrame does not contain a 'Year' column for grouping.")
    #
    #         sheet_name = self.ensure_unique_sheet_name(f"{season} (Predicted Data)" )
    #         sheet = self.workbook.add_worksheet(sheet_name)
    #
    #         # Write headers
    #         sheet.write(0, 0, 'Year')
    #         sheet.write(0, 1, 'Temperature (°C)')
    #         sheet.write(0, 2, 'Evaporation (m of water equivalent)')
    #         sheet.write(0, 3, 'Relative Humidity (%)')
    #         sheet.write(0, 4, 'Predicted Precipitation (m)')
    #
    #         row_idx = 1
    #         # Process each year within the current season
    #         for year, year_data in season_df.groupby('Year'):
    #             # Fetch mean and max values for this year in the current season
    #             mean_temp = round(year_data['Temperature'].mean()) #np.mean(year_data['Temperature'])
    #             max_temp = round(np.max(year_data['Temperature']))
    #             mean_humidity = round(year_data['Relative Humidity (%)'].mean()) # np.mean(year_data['Relative Humidity (%)'])
    #             max_humidity = round(np.max(year_data['Relative Humidity (%)']))
    #             mean_evaporation = year_data['Evaporation'].mean() #np.mean(year_data['Evaporation'])
    #             # mean without () -> ERROR:: unsupported operand type(s) for *: 'float' and 'method'
    #             max_evaporation = np.max(year_data['Evaporation'])
    #
    #             # Step 1: Increment temperature from mean to max, keep evaporation and humidity constant (mean)
    #             for temp in range(int(mean_temp), int(max_temp) + 1):
    #                 predicted_precip = (a1 * temp) + (a2 * mean_humidity) + (a3 * mean_evaporation) + intercept
    #                 predicted_data.append([year, temp, mean_evaporation, mean_humidity, predicted_precip])
    #
    #                 sheet.write(row_idx, 0, year, temp_color_format)
    #                 sheet.write(row_idx, 1, np.clip(temp, mean_temp, 55), temp_color_format)  # Temperature increasing
    #                 sheet.write(row_idx, 2, np.clip(mean_evaporation, mean_evaporation, max_evaporation), temp_color_format)  # Evaporation constant (mean)
    #                 sheet.write(row_idx, 3, np.clip(mean_humidity, mean_humidity, 100), temp_color_format)  # Humidity constant (mean)
    #                 sheet.write(row_idx, 4, predicted_precip, temp_color_format)
    #                 row_idx += 1
    #
    #             # Step 2: Increment evaporation from mean to max, keep temperature and humidity constant (mean)
    #             for evap in np.arange(mean_evaporation, max_evaporation, step_size):
    #                 predicted_precip = (a1 * mean_temp) + (a2 * mean_humidity) + (a3 * evap) + intercept
    #                 predicted_data.append([year, mean_temp, evap, mean_humidity, predicted_precip])
    #
    #                 sheet.write(row_idx, 0, year, evap_color_format)
    #                 sheet.write(row_idx, 1, np.clip(mean_temp, mean_temp, 55), evap_color_format)  # Temperature constant (mean)
    #                 sheet.write(row_idx, 2, np.clip(evap, mean_evaporation, max_evaporation), evap_color_format)  # Evaporation increasing
    #                 sheet.write(row_idx, 3, np.clip(mean_humidity, mean_humidity, 100), evap_color_format)  # Humidity constant (mean)
    #                 sheet.write(row_idx, 4, predicted_precip, evap_color_format)
    #                 row_idx += 1
    #
    #             # Step 3: Increment relative humidity from mean to max, keep temperature and evaporation constant (mean)
    #             for humidity in range(int(mean_humidity), int(max_humidity) + 1):
    #                 predicted_precip = (a1 * mean_temp) + (a2 * humidity) + (a3 * mean_evaporation) + intercept
    #                 predicted_data.append([year, mean_temp, mean_evaporation, humidity, predicted_precip])
    #
    #                 sheet.write(row_idx, 0, year, humidity_color_format)
    #                 sheet.write(row_idx, 1, np.clip(mean_temp, mean_temp, 55), humidity_color_format)  # Temperature constant (mean)
    #                 sheet.write(row_idx, 2, np.clip(mean_evaporation, mean_evaporation, max_evaporation), humidity_color_format)  # Evaporation constant (mean)
    #                 sheet.write(row_idx, 3, np.clip(humidity, mean_humidity, 100), humidity_color_format)  # Humidity increasing
    #                 sheet.write(row_idx, 4, predicted_precip, humidity_color_format)
    #                 row_idx += 1
    #
    #             row_idx += 1  # Move to next row for the next year
    #     print("Predicted data for each season have been written successfully")
    #     return pd.DataFrame(predicted_data, columns=['Year', 'Temperature (°C)', 'Evaporation (m)', 'Humidity (%)', 'Predicted Precipitation (m)'])

    # -----------------------------------------------
    #Aggregated predicted data for each year
    def write_aggregated_predicted_data(self,predicted_df,aggregated_all_seasons):
        # predicted_df = self.write_predicted_data() the code of "write_predicted_data" is written twice
        aggregated_predicted_data = predicted_df.groupby('Year').agg({
            'Temperature (°C)': 'mean',
            'Evaporation (m)': 'mean',
            'Humidity (%)': 'mean',
            'Predicted Precipitation (m)': 'sum'
        }).reset_index()

        # Write Aggregated Predicted Data to a sheet
        sheet_aggregated_predicted = self.workbook.add_worksheet('Aggregated Predicted Data')

        # Define the headers for the aggregated predicted data
        headers = [
            "Year",
            "Temperature (°C)",
            "Evaporation (m)",
            "Relative Humidity (%)",
            "Total Predicted Precipitation (m)"
        ]

        # Write headers to the first row of the new sheet
        for col, header in enumerate(headers):
            sheet_aggregated_predicted.write(0, col, header)

        # Write aggregated predicted data with color formatting
        for row_idx, row in aggregated_predicted_data.iterrows():
            year = row['Year']
            color_format = self.year_colors.get(year, self.workbook.add_format())  # Default format if year is not in year_colors

            # Write the values to the sheet
            for col_idx, value in enumerate(row):
                sheet_aggregated_predicted.write(row_idx + 1, col_idx, value, color_format)

        print("Aggregated predicted data for all seasons have been written successfully ✅")
        print("")
        plot_predicted_weather_data(aggregated_predicted_data, self.workbook)
        plot_combined_weather_data(aggregated_all_seasons,aggregated_predicted_data, self.workbook)

    def close(self):
        self.workbook.close()