
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

#Load Data
# The file was downloaded by kagglehub.dataset_download to the path stored in adorigueto_cnc_turning_roughness_forces_and_tool_wear_path
# Load the 'Exp2.csv' file into a pandas DataFrame
df_exp = pd.read_csv(r"C:\Users\babur\Documents\NITW\MINI PROJECT\DATA SCIENCE\Surface roughness in CNC turning operations using cutting parameters and forces\Exp2.csv")

# Remove unnecessary columns from the DataFrame
df_exp = df_exp.drop(columns=['Run_ID', 'Experiment', 'Replica', 'Tool_ID', 'Group', 'Subgroup',
           'Position', 'Condition',
           'Machined_length', 'Init_diameter', 'Final_diameter', 'CTime', 'R_measurement'])

# Select the input features (X) by dropping the target variable and other roughness related columns
X = df_exp.drop(columns=["Ra", "Rz", "Rsk", "Rku", "RSm", "Rt"])
# Select the target variable (y)
y = df_exp[["Ra"]]

# Display the first 5 rows of the features DataFrame (X)
X.head()

# Display the first 5 rows of the target variable DataFrame (y)
y.head()

from sklearn.preprocessing import MinMaxScaler # Import the MinMaxScaler

scaler = MinMaxScaler() # Create a MinMaxScaler object

scaler.fit(X) # Fit the scaler to the features data to learn the min and max values

# Transform the features data using the fitted scaler
X_scaled = scaler.transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Split the training set into training and validation sets (60% train, 20% validation from the original data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

# Re-running this cell to make variables available

# Print the shapes of the training, validation, and test sets to verify the split
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

from sklearn.tree import DecisionTreeRegressor # Import the DecisionTreeRegressor model
from sklearn.metrics import mean_squared_error # Import mean_squared_error for evaluation
import matplotlib.pyplot as plt # Import matplotlib for plotting

# Create a Decision Tree Regressor model with a specified random state for reproducibility
model = DecisionTreeRegressor(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Get evaluation results (Not applicable for Decision Tree in the same way as XGBoost)
# results = model.evals_result()

# Make predictions on the test set using the trained model
y_pred = model.predict(X_test)

# Create a scatter plot to visualize the true values vs. the predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values') # Label the x-axis as 'True Values'
plt.ylabel('Predictions') # Label the y-axis as 'Predictions'
plt.axis('equal') # Set the scaling of the axes to be equal
plt.axis('square') # Set the aspect ratio of the plot to be square
plt.xlim([0,plt.xlim()[1]]) # Set the x-axis limits starting from 0
plt.ylim([0,plt.ylim()[1]]) # Set the y-axis limits starting from 0
_ = plt.plot([-100, 100], [-100, 100]) # Plot a diagonal line for reference (where true values equal predictions)

from math import sqrt # Import the sqrt function from the math module
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error # Import evaluation metrics

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)
r2 = np.around(r2, 2) # Round the R-squared score to 2 decimal places
print('r2 Score:', r2) # Print the R-squared score

# Calculate the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
mae = np.around(mae, 3) # Round the MAE to 3 decimal places
print('MAE:', mae) # Print the MAE

# Calculate the Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y_test, y_pred)
mape = np.around(mape, 3) # Round the MAPE to 3 decimal places
print('MAPE:', mape) # Print the MAPE

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)  # Use squared=True or default to calculate MSE
mse = np.around(mse, 3) # Round the MSE to 3 decimal places
print('MSE:', mse) # Print the MSE

# Calculate the Root Mean Squared Error (RMSE)
rmse = sqrt(mean_squared_error(y_test, y_pred))
rmse = np.around(rmse, 3) # Round the RMSE to 3 decimal places
print('RMSE:', rmse) # Print the RMSE

# Select one sample from the test set
sample_index = 0  # You can change this index to select a different sample
sample_X = X_test[sample_index]
sample_y = y_test.iloc[sample_index]

# Reshape the sample to be a 2D array as expected by the predict method
sample_X = sample_X.reshape(1, -1)

# Make a prediction for the selected sample
predicted_y = model.predict(sample_X)

# Display the actual and predicted values
print(f"Actual Ra: {sample_y.values[0]}")
print(f"Predicted Ra: {predicted_y[0]}")

# Get the feature importances from the trained model
feature_importances = model.feature_importances_

# Create a list of feature names
feature_names = ['TCond', 'ap', 'vc', 'f', 'Fx', 'Fy', 'Fz', 'F']  # Assuming X is your feature data DataFrame

# Create a DataFrame to store feature names and their importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize the feature importances
plt.figure(figsize=(10, 6)) # Create a figure with a specified size
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance']) # Create a horizontal bar plot
plt.xlabel('Feature Importance') # Label the x-axis
plt.ylabel('Feature') # Label the y-axis
plt.title('Feature Importance Plot (Decision Tree)') # Set the title of the plot
plt.show() # Display the plot


