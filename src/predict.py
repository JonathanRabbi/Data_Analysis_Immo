import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score




def evaluate_xgboost_regression(data_file_path,data):
    df = pd.read_csv(data_file_path)

    
    df_House = df[df['Type of property'] == 'house']
    X = df_House[['Bedrooms', 'Bathrooms', 'Toilets', 'Surface of the plot']]
    y = df_House['Price']

     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=5)

    xg_reg = XGBRegressor()
    xg_reg.fit(X_train, y_train)

    
    y_pred = xg_reg.predict(X_test)

    input_data = np.array([[data.Bedrooms, data.Bathrooms, data.Toilets, data.Surface_of_plot]])
    predicted_price = xg_reg.predict(input_data)

    predicted = float(predicted_price[0])
    valuation = f'{round(predicted,2)} Euro'

    return valuation