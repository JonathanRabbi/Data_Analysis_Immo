import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def evaluate_regression_model(model_type, data_file_path):
    # Load data from the CSV file
    df = pd.read_csv(data_file_path)

    # Filter data for houses and prepare feature and target
    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot']]
    y = df_House['Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

    if model_type == "XGBoost":
        # Create and train the XGBoost regressor
        model = XGBRegressor()
    elif model_type == "Decision Tree":
        # Create and train the Decision Tree regressor
        model = DecisionTreeRegressor(criterion='mse', max_depth=10, min_samples_split=10)
    else:
        raise ValueError("Invalid model type. Please choose 'XGBoost' or 'Decision Tree'.")

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print the RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'{model_type} - RMSE_trained_model: {rmse:.2f}')

    # Perform cross-validation to evaluate the model
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_cv = np.sqrt(-scores)
    print(f'{model_type} - Cross-validated RMSE: {rmse_cv.mean():.2f}')

    # Print the model scores on training and testing data
    print(f'{model_type} - Training set R^2 score: {model.score(X_train, y_train):.2f}')
    print(f'{model_type} - Testing set R^2 score: {model.score(X_test, y_test):.2f}')

if __name__ == "__main__":
    data_file_path = '/Users/jonathanrabbi/Desktop/Data_Analysis_Immo/CSV_Files/Cat_House_details.csv'
    evaluate_regression_model("XGBoost", data_file_path)
    
    evaluate_regression_model("Decision Tree", data_file_path)
