from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel
from src.predict import evaluate_xgboost_regression

app=FastAPI()

class House_feature(BaseModel):
    Bedrooms: int
    Bathrooms: int
    Toilets: int
    Surface_of_plot: float


@app.get("/")
def activate():
    return ("Welcome, please enter to have your house valued")

@app.post("/predict-house")
def predict_house_price(data:House_feature):
    data_file_path='/Users/jonathanrabbi/Desktop/Data_Analysis_Immo/Data/Cat_House_details.csv'
    try:
        result=evaluate_xgboost_regression(data_file_path,data)
        return{"valued at":result}
    except ValueError: # if the user does not insert numbers
        raise HTTPException(status_code=400, detail="please insert numbers.")
    
    

    