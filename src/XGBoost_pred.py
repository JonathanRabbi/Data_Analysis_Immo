import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score




def evaluate_xgboost_regression(data_file_path,data):
    # create dataframe from the csv file that is opened
    df = pd.read_csv(data_file_path)

    # Only taking data for houses and preparing etermining the feature and target
    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot']]
    y = df_House['Price']

    # Splitting the data to train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5)

    # creating and training the XGB regressor. Here the regression model has been fitted on training data
    xg_reg = XGBRegressor()
    xg_reg.fit(X_train, y_train)

    # Based on trained model the below code conveys that the regression model of the trained data would be adapted to the test data
    y_pred = xg_reg.predict(X_test)

    # The MSE conveys the performance of the predicted value based on the model and the actual target values (y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    #print('MSE: %0.2f' % mse)
    #print('RMSE_trained_model: $%0.2f' % rmse)

    '''Cross Validation is used to evaluate how well the model performs with unseen data. 
        Here the rmse is calculated for different subsets and how consistent it is'''
    
    model = XGBRegressor(objective='reg:squarederror')

    #In order for the prediction to take place
    input_data = np.array([[data.Bedrooms, data.Bathrooms, data.Toilets, data.Surface_of_plot]])
    predicted_price = xg_reg.predict(input_data)

    predicted = float(predicted_price[0])
    valuation = f'{round(predicted,2)} Euro'

    return valuation