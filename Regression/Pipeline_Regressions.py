#!/usr/bin/env python3

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

def evaluate_xgboost_regression(data_file_path):
    # create dataframe from the csv file that is opened
    df = pd.read_csv(data_file_path)

    # Only taking data for houses and preparing etermining the feature and target
    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot','Building_encoded']]
    y = df_House['Price']

    # Splitting the data to train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

    # creating and training the XGB regressor. Here the regression model has been fitted on training data
    xg_reg = XGBRegressor()
    xg_reg.fit(X_train, y_train)

    # Based on trained model the below code conveys that the regression model of the trained data would be adapted to the test data
    y_pred = xg_reg.predict(X_test)

    # The MSE conveys the performance of the predicted value based on the model and the actual target values (y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    #print('MSE: %0.2f' % mse)
    print('RMSE_trained_model: $%0.2f' % rmse)

    '''Cross Validation is used to evaluate how well the model performs with unseen data. 
        Here the rmse is calculated for different subsets and how consistent it is'''
    
    model = XGBRegressor(objective='reg:squarederror')
    scores = cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=10)
    rmse_cv = np.sqrt(-scores)
    print('XGBoost-Cross-validated RMSE: %0.2f' % rmse_cv.mean())

    # displaying the scores (R^2) of the model from trained data and test set.
    print('XGBoost-Training set R^2 score: %0.2f' % xg_reg.score(X_train, y_train))
    print('XGBoost-Testing set R^2 score: %0.2f' % xg_reg.score(X_test, y_test))



def evaluate_decisiontree_regression(data_file_path):
    df=pd.read_csv(data_file_path)

    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot','Building_encoded']]
    y = df_House['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

    reg_tree=DecisionTreeRegressor(criterion='squared_error',
                          max_depth=10,
                          min_samples_split=10,
                          )
    
    reg_tree.fit(X_train,y_train)

    y_pred=reg_tree.predict(X_test)

    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    print('Decision Tree-RMSE_trained_model: $%0.2f' % rmse)

    cross_val_score(reg_tree,X_train, y_train, cv=10)

    print('Decision Tree-Training set R^2 score: %0.2f' % reg_tree.score(X_train, y_train))
    print('Decision Tree-Testing set R^2 score: %0.2f' % reg_tree.score(X_test, y_test))



def evaluate_linear_regression(data_file_path):
    df=pd.read_csv(data_file_path)

    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot','Building_encoded']]
    y = df_House['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=0)

    reg_linear= LinearRegression()
    reg_linear.fit(X_train,y_train)

    y_pred=reg_linear.predict(X_test)

    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)

    print('Linear Regression-RMSE_trained_model: $%0.2f' % rmse)

    cross_val_score(reg_linear,X_train, y_train, cv=10)

    print('Linear Regression -Training set R^2 score: %0.2f' % reg_linear.score(X_train, y_train))
    print('Linear Regression -Testing set R^2 score: %0.2f' % reg_linear.score(X_test, y_test))



if __name__ == "__main__":
    data_file_path = '/Users/jonathanrabbi/Desktop/Data_Analysis_Immo/CSV_File_Accomodation/Cat_House_details.csv'
    evaluate_xgboost_regression(data_file_path)
    evaluate_decisiontree_regression(data_file_path)
    evaluate_linear_regression(data_file_path)
