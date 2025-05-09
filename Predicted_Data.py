import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PrecipitationPredictorAI:
    def __init__(self, workbook, detailed_df, seasons_data):
        self.workbook = workbook
        self.detailed_df = detailed_df
        self.seasons_data = seasons_data

    def fit_ai_model(self): #version 1 before testing
        filtered_df = self.detailed_df.dropna(subset=['Temperature', 'Relative Humidity (%)', 'Evaporation', 'Precipitation'])

        X = filtered_df[['Temperature', 'Relative Humidity (%)', 'Evaporation']]
        y = filtered_df['Precipitation']

        X_train, X_test, y_train, y_test = train_test_split(X, y, 	test_size=0.05, random_state=42)

        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4,],
            'max_features': ['sqrt', 'log2'],
        }

        model_rf = RandomForestRegressor(random_state=42)
        grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)#verbose prints all progress and status updates
        grid_search_rf.fit(X_train, y_train)
        best_model = grid_search_rf.best_estimator_

        best_model.fit(X, y)

        return best_model, X_test, y_test

    # Write predicted data based on modified values using AI model
    def write_predicted_data(self):
        # Define color formats for temperature, evaporation, and humidity changes
        temp_color_format = self.workbook.add_format({'bg_color': '#FFC7CE'})
        evap_color_format = self.workbook.add_format({'bg_color': '#C6EFCE'})
        humidity_color_format = self.workbook.add_format({'bg_color': '#FFEB9C'})
        step_size = 0.00001  # Small step size for evaporation increments

        # Train the AI model
        model,_,_ = self.fit_ai_model()

        # Initialize an empty list to store predicted data
        predicted_data = []

        # Step 3: Iterate through seasons and predict based on modified data
        for season, season_data in self.seasons_data.items():
            # Convert the season_data (which is a list) into a DataFrame
            season_df = pd.DataFrame(season_data)

            # Ensure the DataFrame has a 'Year' column
            if 'Year' not in season_df.columns:
                raise ValueError("The DataFrame does not contain a 'Year' column for grouping.")

            # Create a new worksheet for the predicted data
            sheet_name = f"{season} (Predicted Data)"
            sheet = self.workbook.add_worksheet(sheet_name)

            # Write headers in the Excel sheet
            sheet.write(0, 0, 'Year')
            sheet.write(0, 1, 'Temperature (°C)')
            sheet.write(0, 2, 'Evaporation (m)')
            sheet.write(0, 3, 'Humidity (%)')
            sheet.write(0, 4, 'Predicted Precipitation (m)')

            row_idx = 1  # Start writing at row 1

            # Process each year's data
            for year, year_data in season_df.groupby('Year'):
                # Add a blank row to separate years visually
                if row_idx > 1:
                    for column in range(0,5):
                        sheet.write(row_idx, column, '-------------')
                    row_idx += 1

                # Calculate mean and max values for each year
                mean_temp = year_data['Temperature'].mean()
                max_temp =year_data['Temperature'].max()
                mean_humidity = year_data['Relative Humidity (%)'].mean()
                max_humidity =year_data['Relative Humidity (%)'].max()
                mean_evaporation = year_data['Evaporation'].mean()
                max_evaporation = year_data['Evaporation'].max()

                # Increment temperature, keeping evaporation and humidity constant
                for temp in np.arange(mean_temp, max_temp + 1,step= .4):
                    # Predict precipitation using the trained AI model
                    X_pred = pd.DataFrame([[temp, mean_humidity,mean_evaporation]],
                                          columns=['Temperature', 'Relative Humidity (%)', 'Evaporation'])

                    predicted_precip = model.predict(X_pred)[0]

                    predicted_data.append([year, temp, mean_evaporation, 				mean_humidity, predicted_precip])

                    # Write predicted data to Excel
                    sheet.write(row_idx, 0, year, temp_color_format)
                    sheet.write(row_idx, 1, temp, temp_color_format)
                    sheet.write(row_idx, 2, mean_evaporation, temp_color_format)
                    sheet.write(row_idx, 3, mean_humidity, temp_color_format)
                    sheet.write(row_idx, 4, predicted_precip, temp_color_format)
                    row_idx += 1

                for evap in np.arange(mean_evaporation, max_evaporation,step_size):

                    X_pred = pd.DataFrame([[temp, mean_humidity,mean_evaporation]],
                                          columns=['Temperature', 'Relative Humidity (%)', 'Evaporation'])
                    predicted_precip = model.predict(X_pred)[0]

                    predicted_data.append([year, mean_temp, evap, mean_humidity, predicted_precip])

                    # Write predicted data to Excel
                    sheet.write(row_idx, 0, year, evap_color_format)
                    sheet.write(row_idx, 1, mean_temp, evap_color_format)
                    sheet.write(row_idx, 2, evap, evap_color_format)
                    sheet.write(row_idx, 3, mean_humidity, evap_color_format)
                    sheet.write(row_idx, 4, predicted_precip, evap_color_format)
                    row_idx += 1


                for humidity in np.arange(mean_humidity, max_humidity + 1,step=.4):

                    X_pred = pd.DataFrame([[temp, mean_humidity,mean_evaporation]],
                                          columns=['Temperature', 'Relative Humidity (%)', 'Evaporation'])
                    predicted_precip = model.predict(X_pred)[0]


                    predicted_data.append([year, mean_temp, mean_evaporation,humidity, predicted_precip])

                    # Write predicted data to Excel
                    sheet.write(row_idx, 0, year, humidity_color_format)
                    sheet.write(row_idx, 1, mean_temp, humidity_color_format)
                    sheet.write(row_idx, 2, mean_evaporation,humidity_color_format)
                    sheet.write(row_idx, 3, humidity, humidity_color_format)
                    sheet.write(row_idx, 4, predicted_precip,humidity_color_format)
                    row_idx += 1
            row_idx += 1  # Move to the next row for the next year

        print("Predicted data for each season have been written successfully ✅")

        self.evaluate_model()
        # Return predicted data in a DataFrame
        return pd.DataFrame(predicted_data, columns=['Year', 'Temperature (°C)', 'Evaporation (m)', 'Humidity (%)', 'Predicted Precipitation (m)'])

    # Test the model and evaluate its accuracy
    def evaluate_model(self):
        model, X_test, y_test= self.fit_ai_model()

        # Make predictions on the test set using a pandas DataFrame for X_test
        X_test_df = pd.DataFrame(X_test, columns=['Temperature', 'Relative Humidity (%)', 'Evaporation'])

        # Make predictions on the test set
        y_pred = model.predict(X_test_df)

        # Calculate evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        accuracy_percentage = round(r2*100,1)

        # Print the evaluation results
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R² Score (R2): {r2}")
        print(f"Accuracy Percentage: {accuracy_percentage}%")

# ====================================
# trial version ,but it has problems

# def fit_ai_model(self):
#
#     # Filter rows where the necessary data is available
#     filtered_df = self.detailed_df.dropna(subset=['Temperature', 'Relative Humidity (%)', 'Evaporation', 'Precipitation'])
#
#     # Define features (X) and target (y)
#     X = filtered_df[['Temperature', 'Relative Humidity (%)', 'Evaporation']]
#     y = filtered_df['Precipitation']
#
#     # Apply log transformation to the target variable if needed
#     y_log = np.log1p(y)
#     # y_log = np.log1p(self.detailed_df['Precipitation'])  # Adding 1 to avoid log(0)
#
#     # Feature Scaling using StandardScaler
#     # scaler = StandardScaler() > import StandardScaler ::trying to solve the problem of predicted precipitation is the same valuer that Scales the features to have a mean of 0 and a standard deviation of 1. However, sometimes over-scaling can cause the model to struggle in differentiating between different input values, especially if the features do not have a strong correlation with the target variable.
#     scaler = MinMaxScaler()
#     X_scaled = scaler.fit_transform(X)
#     X_scaled = pd.DataFrame(X_scaled, columns=['Temperature', 'Relative Humidity (%)', 'Evaporation'])  # Preserve column names
#
#     # Feature Engineering - Add Polynomial Features and Interaction Terms
#     poly = PolynomialFeatures(degree=2, interaction_only=True)
#     X_poly = poly.fit_transform(X_scaled)
#
#     # Split the data into training and testing sets (80% training, 20% testing)
#     X_train, X_test, y_train, y_test = train_test_split(X_poly, y_log, test_size=0.2, random_state=42)
#
#     # Hyperparameter Tuning using GridSearchCV
#     # Random Forest may sometimes predict constant values if it's not properly tuned or the data isn't diverse enough.
#     # param_grid = {
#     #     'n_estimators': [500, 1000],# Try more trees
#     #     'max_depth': [10, 20, 50], # Experiment with deeper trees
#     #     'min_samples_split': [2, 5, 10],
#     #     'min_samples_leaf': [1, 2, 4],
#     # }
#     #  The model might need more extensive hyperparameter tuning
#     param_grid = {
#         'n_estimators': [100, 200, 500, 1000],
#         'max_depth': [10, 20, 30, 50, None],
#         'min_samples_split': [2, 5, 10, 20],
#         'min_samples_leaf': [1, 2, 4, 6],
#         'max_features': ['sqrt', 'log2'],
#     }
#
#     model_rf = RandomForestRegressor(random_state=42)
#     grid_search_rf = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)#verbose prints all progress and status updates
#     grid_search_rf.fit(X_train, y_train)
#     best_model = grid_search_rf.best_estimator_
#     model_rf.fit(X_train, y_train)
#     # Fit the best model on the entire dataset
#     best_model.fit(X_scaled, y)
#
#     # # LightGBM Model
#     # model_lgb = lgb.LGBMRegressor(n_estimators=1000, random_state=42)
#     # model_lgb.fit(X_train, y_train)
#     #
#     # # Ensemble model using Voting Regressor
#     # ensemble_model = VotingRegressor(estimators=[('rf', grid_search_rf.best_estimator_), ('lgb', model_lgb)])
#     #
#     # # Fit the ensemble model on the entire dataset
#     # ensemble_model.fit(X_poly, y)
#
#     # Check feature importance after fitting the model to know how much each feature contributes to the prediction.
#     # foe ex Evaporation: This feature has the most influence on the model's predictions, suggesting that changes in evaporation have a strong impact on precipitation predictions.
#     # importances = best_model.feature_importances_
#     # print(f"{importances}")
#
#
#     # # Provides a more reliable and generalizable evaluation of your model's performance instead of just relying on a single train/test spli, helping you avoid overfitting and making sure the model performs consistently across different subsets of the data. It’s especially helpful when tuning hyperparameters and when you want a more robust estimate of model performance.
#     # cross_val_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
#
#     # xgb_model = XGBRegressor(n_estimators=100, random_state=42)
#     #
#     # # Create an ensemble model using Voting Regressor
#     # ensemble_model = VotingRegressor(estimators=[('rf', best_model), ('xgb', xgb_model)])
#     # # Fit the ensemble model on the training data
#     # ensemble_model.fit(X_train, y_train)
#     #
#     # # Fit the ensemble model on the entire dataset
#     # ensemble_model.fit(X_scaled, y)
#
#     return best_model, X_test, y_test
