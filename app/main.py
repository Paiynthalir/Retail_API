# Import Libraries
from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd
from datetime import date
from typing import Dict, List

#  Instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

# Load the chosen prediction and forcasting trained models from models folder 
# and save it into variables

sgd_pipe = load('models/Predictive/sgd_pipeline.joblib')
prophet_model = load('models/Forecasting/Prof_model1.joblib')

# create a function called read_root() 
# nd Add a decorator to it in order to add a GET endpoint to app on the root
@app.get("/")
def read_root():
    return {
        "description": "This API provides two models deployed in production: "
                       "1. A predictive model using a Machine Learning algorithm to predict sales revenue for a given item in a specific store on a given date. "
                       "2. A forecasting model using a time-series analysis algorithm to forecast total sales revenue across all stores and items for the next 7 days.",
        "endpoints": {
            "/": "Brief description of the project objectives",
            "/health/": "Returns a status code 200 with a welcome message",
            "/sales/national/": "Forecasts total sales revenue for the next 7 days.",
            "/sales/stores/items/": "Predicts sales revenue for a specific store and item."
        },
        "expected_input_parameters": {
            "date": "YYYY-MM-DD format",
            "store_id": "Identifier of the store",
            "item_id": "Identifier of the item"
        },
        "output_format": {
            "national": {
                "date": "sales_amount"
            },
            "stores_items": {
                "prediction": "sales_amount"
            }
        },
        "github_repo": "https://github.com/Paiynthalir/Retail_API"
    }

# get health endpoint
@app.get('/health', status_code=200)
def healthcheck():
    return {"message": "Hi there! Hearty Welcome to Sales Prediction API"}

# Function to extract the features from the input and this output can be the input for the model prediction 
def extract_features(item_id: str, store_id: str, date_str: str) -> Dict[str, List]:
    # Convert the string date to a date object
    parsed_date = date.fromisoformat(date_str)
    
    # Extract features
    # Extract dept_id from item_id
    # Split the item_id by underscores and retrieve the relevant part
    parts = item_id.split("_")
    if len(parts) > 1:
        # Combine the first two parts to form dept_id
        dept_id = "_".join(parts[:2])

    # Extract date features 
    day_name = parsed_date.strftime('%a')  # Abbreviated day name
    month_name = parsed_date.strftime('%b')  # Abbreviated month name
    year = str(parsed_date.year)  # Year as a string
    
    # Return the features in a dictionary
    features = {
        "dept_id": [dept_id],
        "store_id": [store_id],
        "day_name": [day_name],
        "month": [month_name],
        "year": [year],
    }
    
    return features

class SalesForecastRequest(BaseModel):
    date: date

@app.get("/sales/national/", response_model=Dict[str, float])
def forecast_sales(request: SalesForecastRequest):
    # Here you would implement your forecasting logic
    # Placeholder implementation
    return {
        "2016-01-01": 10000.01,
        "2016-01-02": 10001.12,
        "2016-01-03": 10002.22,
        "2016-01-04": 10003.30,
        "2016-01-05": 10004.46,
        "2016-01-06": 10005.12,
        "2016-01-07": 10006.55,
    }

class SalesPredictionRequest(BaseModel):
    date: date
    store_id: int
    item_id: int

@app.get("/sales/stores/items/", response_model=Dict[str, float])
def predict_sales(request: SalesPredictionRequest):
    # Here you would implement your sales prediction logic
    # Placeholder implementation
    return {"prediction": 19.72}
