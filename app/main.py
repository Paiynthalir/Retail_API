# Import Libraries
from fastapi import FastAPI
from starlette.responses import JSONResponse
from fastapi.responses import JSONResponse
from fastapi import HTTPException
import json
from joblib import load
import pandas as pd
from datetime import date
from datetime import datetime, timedelta
from typing import Dict, List
import re


#  Instantiate a FastAPI() class and save it into a variable called app
app = FastAPI()

# Load the chosen prediction and forcasting trained models from models folder 
# and save it into variables

xgb_pipe = load('models/Predictive/xgb_pipeline.joblib')
prophet_model = load('models/Forecasting/Prof_model1.joblib')

# create a function called read_root() 
# nd Add a decorator to it in order to add a GET endpoint to app on the root
@app.get("/")
def read_root():
    response_data = {
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
    
    # Pretty-print the JSON response
    return JSONResponse(content=json.loads(json.dumps(response_data, indent=4)))

# get health endpoint
@app.get('/health', status_code=200)
def healthcheck():
    return {"message": "Hi there! Welcome to Sales Prediction API"}


# Function to extract the features from the input and this output can be the input for the model prediction 
def extract_features(item_id: str, store_id: str, date_str: str) -> Dict[str, List]:
    # Convert the string date to a date object
    try:
        parsed_date = date.fromisoformat(date_str)
    except ValueError:
        raise ValueError("Invalid date format. Use YYYY-mm-dd.")
    
    # Define the regex pattern for valid item_id
    pattern = r"^[A-Z]+_\d_[0-9]{3}$"  

    # Split the item_id by underscores
    parts = item_id.split("_")

    # Validate the item_id format
    if len(parts) == 3 and re.match(pattern, item_id):
        # Combine the first two parts to form dept_id
        dept_id = "_".join(parts[:2])
    else:
        raise ValueError("Invalid item ID format. Expected format is <characters>_<singledigit>_<3digits>.")
    
    # Define the regex pattern for valid store_id
    pattern = r"^(CA|TX|WI)_\d+$"
    # Validate store_id format
    if re.match(pattern, store_id):
        store_number = store_id.split("_")[1]  
    else:
        raise ValueError("Invalid store ID. Expected values CA_#, TX_#, and WI_#")


    # Extract date features 
    day_name = parsed_date.strftime('%a')  # Abbreviated day name
    month_name = parsed_date.strftime('%b')  # Abbreviated month name
    year = str(parsed_date.year)  # Year as a string
    
    # Return the features in a dictionary
    features = {
        "dept_id": [dept_id],
        "store_id": [store_id],
        "day_name": [day_name],
        "month_name": [month_name],
        "year": [year]
    }
    
    return features


@app.get("/sales/stores/items/", response_model=Dict[str, float])
def predict_sales(item_id: str, store_id: str, date: str) -> Dict[str, float]:
    try:
        # Extraction of features from the input parameters
        features = extract_features(item_id, store_id, date)
        # Convert the input into a pandas DataFrame
        obs = pd.DataFrame(features)

        # Predict the sales revenue for the given item and store on a given date 
        pred = xgb_pipe.predict(obs)[0]  

        # Return the prediction in the specified JSON format
        return {"prediction": round(float(pred), 2)} 
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Function to call the saved forecast model
@app.get("/sales/national/", response_model=Dict[str, float])
def forecast_sales(date: str) -> Dict:
    try:
        # Validate and parse the input date
        parsed_date = datetime.strptime(date, "%Y-%m-%d")

        # Create a DataFrame for the next 7 days
        future_dates = pd.date_range(start=parsed_date + timedelta(days=1), periods=7)
        future_df = pd.DataFrame({'ds': future_dates})

        # Make predictions using the Prophet model
        forecast = prophet_model.predict(future_df)

        # Prepare the forecasted sales data in the specified JSON format
        forecasted_sales = {row['ds'].strftime('%Y-%m-%d'): round(row['yhat'], 2) for _, row in forecast.iterrows()}

        return forecasted_sales

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))