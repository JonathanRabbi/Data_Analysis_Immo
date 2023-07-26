from fastapi import FastAPI, Path
from pydantic import BaseModel
from src.XGBoost_pred import evaluate_xgboost_regression

app=FastAPI()

class House_feature(BaseModel):
    Bedrooms: int
    Bathrooms: int
    Toilets: int
    Surface_of_plot: float


@app.get("/")
def activate():
    return "Welcome, please enter to have your house valued"

@app.post("/predict-house")
def predict_house_price(data:House_feature):
    data_file_path='/Users/jonathanrabbi/Desktop/Data_Analysis_Immo/Data/Cat_House_details.csv'
    result=evaluate_xgboost_regression(data_file_path,data)
    return{"valued at":result}
    