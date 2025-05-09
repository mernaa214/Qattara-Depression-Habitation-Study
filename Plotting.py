import matplotlib
matplotlib.use('Agg')  # "Non-GUI backend" Anti-Grain Geometry (AGG)-> Backend in Matplotlib used for generating plots in non-interactive environments
import matplotlib.pyplot as plt

# Function for generating and saving plots
def plot_weather_data(aggregated_data,workbook):
    # Generate matplotlib plots
    fig_temp, axs_temp = plt.subplots(1, 1, figsize=(10, 6))
    fig_evap, axs_evap = plt.subplots(1, 1, figsize=(10, 6))
    fig_hum, axs_hum = plt.subplots(1, 1, figsize=(10, 6))

    years = aggregated_data['Year'].values
    temperature_values = aggregated_data['Temperature'].values
    evaporation_values = aggregated_data['Evaporation'].values
    humidity_values = aggregated_data['Relative Humidity (%)'].values
    precipitation_values = aggregated_data['Precipitation'].values

    # Temperature vs Precipitation
    axs_temp.scatter(temperature_values, precipitation_values, c=years, cmap='viridis')
    for i, txt in enumerate(years):
        axs_temp.annotate(txt, (temperature_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    axs_temp.set_title('Temperature vs Precipitation')
    axs_temp.set_xlabel('Temperature (°C)')
    axs_temp.set_ylabel('Precipitation (mm)')
    axs_temp.grid(True)

    # Evaporation vs Precipitation
    axs_evap.scatter(evaporation_values, precipitation_values, c=years, cmap='viridis')
    for i, txt in enumerate(years):
        axs_evap.annotate(txt, (evaporation_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    axs_evap.set_title('Evaporation vs Precipitation')
    axs_evap.set_xlabel('Evaporation (mm)')
    axs_evap.set_ylabel('Precipitation (mm)')
    axs_evap.grid(True)

    # Humidity vs Precipitation
    axs_hum.scatter(humidity_values, precipitation_values, c=years, cmap='viridis')
    for i, txt in enumerate(years):
        axs_hum.annotate(txt, (humidity_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    axs_hum.set_title('Humidity vs Precipitation')
    axs_hum.set_xlabel('Humidity (%)')
    axs_hum.set_ylabel('Precipitation (mm)')
    axs_hum.grid(True)

    plt.tight_layout()

    # Save plots as images
    plot_file_temp = 'temperature_vs_precipitation.png'
    plot_file_evap = 'evaporation_vs_precipitation.png'
    plot_file_hum = 'humidity_vs_precipitation.png'

    fig_temp.savefig(plot_file_temp)
    fig_evap.savefig(plot_file_evap)
    fig_hum.savefig(plot_file_hum)

    plt.close(fig_temp)
    plt.close(fig_evap)
    plt.close(fig_hum)

    # Insert images side by side in the Excel sheet
    sheet_plots = workbook.add_worksheet('Plots')
    sheet_plots.insert_image('A1', plot_file_temp, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_plots.insert_image('I1', plot_file_evap, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_plots.insert_image('Q1', plot_file_hum, {'x_scale': 0.5, 'y_scale': 0.5})

# --------------------------------------------------------------
def plot_predicted_weather_data(predicted_df,workbook):
    # Generate matplotlib plots
    fig_temp, axs_temp = plt.subplots(1, 1, figsize=(10, 6))
    fig_evap, axs_evap = plt.subplots(1, 1, figsize=(10, 6))
    fig_hum, axs_hum = plt.subplots(1, 1, figsize=(10, 6))

    years = predicted_df['Year'].values
    temperature_values = predicted_df['Temperature (°C)'].values
    evaporation_values = predicted_df['Evaporation (m)'].values
    humidity_values = predicted_df['Humidity (%)'].values
    precipitation_values = predicted_df['Predicted Precipitation (m)'].values

    # Temperature vs Predicted Precipitation
    axs_temp.scatter(temperature_values, precipitation_values, c=years, cmap='viridis', alpha=0.7)
    for i, txt in enumerate(years):
        axs_temp.annotate(txt, (temperature_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    axs_temp.set_title('Temperature vs Predicted Precipitation')
    axs_temp.set_xlabel('Temperature (°C)')
    axs_temp.set_ylabel('Predicted Precipitation (m)')
    axs_temp.grid(True)

    # Evaporation vs Predicted Precipitation
    axs_evap.scatter(evaporation_values, precipitation_values, c=years, cmap='viridis', alpha=0.7)
    for i, txt in enumerate(years):
        axs_evap.annotate(txt, (evaporation_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    axs_evap.set_title('Evaporation vs Predicted Precipitation')
    axs_evap.set_xlabel('Evaporation (m)')
    axs_evap.set_ylabel('Predicted Precipitation (m)')
    axs_evap.grid(True)

    # Humidity vs Predicted Precipitation
    axs_hum.scatter(humidity_values, precipitation_values, c=years, cmap='viridis', alpha=0.7)
    for i, txt in enumerate(years):
        axs_hum.annotate(txt, (humidity_values[i], precipitation_values[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
    axs_hum.set_title('Humidity vs Predicted Precipitation')
    axs_hum.set_xlabel('Humidity (%)')
    axs_hum.set_ylabel('Predicted Precipitation (m)')
    axs_hum.grid(True)

    plt.tight_layout()

    # Save plots as images
    plot_file_temp = 'predicted_temperature_vs_precipitation.png'
    plot_file_evap = 'predicted_evaporation_vs_precipitation.png'
    plot_file_hum = 'predicted_humidity_vs_precipitation.png'

    fig_temp.savefig(plot_file_temp)
    fig_evap.savefig(plot_file_evap)
    fig_hum.savefig(plot_file_hum)

    plt.close(fig_temp)
    plt.close(fig_evap)
    plt.close(fig_hum)

    # Insert images side by side in the Excel sheet
    sheet_predicted_plots = workbook.add_worksheet('Predicted Data Plots')
    sheet_predicted_plots.insert_image('A1', plot_file_temp, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_predicted_plots.insert_image('I1', plot_file_evap, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_predicted_plots.insert_image('Q1', plot_file_hum, {'x_scale': 0.5, 'y_scale': 0.5})

# --------------------------------------------------------------
def plot_combined_weather_data(aggregated_data, predicted_df,workbook):
    # Create subplots for each chart
    fig_temp, axs_temp = plt.subplots(1, 1, figsize=(10, 6))
    fig_evap, axs_evap = plt.subplots(1, 1, figsize=(10, 6))
    fig_hum, axs_hum = plt.subplots(1, 1, figsize=(10, 6))

    # Extract values from both datasets
    years_raw = aggregated_data['Year'].values
    years_pred = predicted_df['Year'].values

    temperature_values_raw = aggregated_data['Temperature'].values
    temperature_values_pred = predicted_df['Temperature (°C)'].values

    evaporation_values_raw = aggregated_data['Evaporation'].values
    evaporation_values_pred = predicted_df['Evaporation (m)'].values

    humidity_values_raw = aggregated_data['Relative Humidity (%)'].values
    humidity_values_pred = predicted_df['Humidity (%)'].values

    precipitation_values_raw = aggregated_data['Precipitation'].values
    precipitation_values_pred = predicted_df['Predicted Precipitation (m)'].values

    # Temperature vs Precipitation: Raw and Predicted
    axs_temp.scatter(temperature_values_raw, precipitation_values_raw, c='blue', label='Raw Data', alpha=0.7)
    axs_temp.scatter(temperature_values_pred, precipitation_values_pred, c='red', label='Predicted Data', alpha=0.7)
    axs_temp.set_title('Temperature vs Precipitation')
    axs_temp.set_xlabel('Temperature (°C)')
    axs_temp.set_ylabel('Precipitation (mm)')

    # Add year annotations for raw data
    for i in range(len(years_raw)):
        axs_temp.annotate(years_raw[i], (temperature_values_raw[i], precipitation_values_raw[i]),
                          textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    # Add year annotations for predicted data
    for i in range(len(years_pred)):
        axs_temp.annotate(years_pred[i], (temperature_values_pred[i], precipitation_values_pred[i]),
                          textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    axs_temp.legend(loc='upper right')
    axs_temp.grid(True)

    # Evaporation vs Precipitation: Raw and Predicted
    axs_evap.scatter(evaporation_values_raw, precipitation_values_raw, c='blue', label='Raw Data', alpha=0.7)
    axs_evap.scatter(evaporation_values_pred, precipitation_values_pred, c='red', label='Predicted Data', alpha=0.7)
    axs_evap.set_title('Evaporation vs Precipitation')
    axs_evap.set_xlabel('Evaporation (mm)')
    axs_evap.set_ylabel('Precipitation (mm)')

    # Add year annotations for raw data
    for i in range(len(years_raw)):
        axs_evap.annotate(years_raw[i], (evaporation_values_raw[i], precipitation_values_raw[i]),
                          textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    # Add year annotations for predicted data
    for i in range(len(years_pred)):
        axs_evap.annotate(years_pred[i], (evaporation_values_pred[i], precipitation_values_pred[i]),
                          textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    axs_evap.legend(loc='upper right')
    axs_evap.grid(True)

    # Humidity vs Precipitation: Raw and Predicted
    axs_hum.scatter(humidity_values_raw, precipitation_values_raw, c='blue', label='Raw Data', alpha=0.7)
    axs_hum.scatter(humidity_values_pred, precipitation_values_pred, c='red', label='Predicted Data', alpha=0.7)
    axs_hum.set_title('Humidity vs Precipitation')
    axs_hum.set_xlabel('Humidity (%)')
    axs_hum.set_ylabel('Precipitation (mm)')

    # Add year annotations for raw data
    for i in range(len(years_raw)):
        axs_hum.annotate(years_raw[i], (humidity_values_raw[i], precipitation_values_raw[i]),
                         textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    # Add year annotations for predicted data
    for i in range(len(years_pred)):
        axs_hum.annotate(years_pred[i], (humidity_values_pred[i], precipitation_values_pred[i]),
                         textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

    axs_hum.legend(loc='upper right')
    axs_hum.grid(True)

    plt.tight_layout()

    # Save plots as images
    plot_file_temp = 'combined_temperature_vs_precipitation.png'
    plot_file_evap = 'combined_evaporation_vs_precipitation.png'
    plot_file_hum = 'combined_humidity_vs_precipitation.png'

    fig_temp.savefig(plot_file_temp)
    fig_evap.savefig(plot_file_evap)
    fig_hum.savefig(plot_file_hum)

    plt.close(fig_temp)
    plt.close(fig_evap)
    plt.close(fig_hum)

    # Insert images side by side in the new Excel sheet
    sheet_combined_plots =workbook.add_worksheet('Combined Plots')
    sheet_combined_plots.insert_image('A1', plot_file_temp, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_combined_plots.insert_image('I1', plot_file_evap, {'x_scale': 0.5, 'y_scale': 0.5})
    sheet_combined_plots.insert_image('Q1', plot_file_hum, {'x_scale': 0.5, 'y_scale': 0.5})