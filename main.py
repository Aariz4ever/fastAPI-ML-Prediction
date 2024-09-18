from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np
import json

# Create FastAPI instance
app = FastAPI()

# Define the input data model
class DiabetesInput(BaseModel):
    Glucose: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    Intercept: float

# Load the saved model
diabetes_model = load('model.joblib')

@app.post("/predict_diabetes")
def predict_diabetes(input_data: DiabetesInput):
    # Convert the input data to a list (array-like)
    input_list = [
        input_data.Glucose, 
        input_data.BMI, 
        input_data.DiabetesPedigreeFunction, 
        input_data.Age, 
        input_data.Intercept
    ]

    # Convert to a NumPy array and reshape it for prediction
    input_array = np.array(input_list).reshape(1, -1)
    
    # Make prediction
    prediction = diabetes_model.predict(input_array)

    # Return the prediction result (0 or 1)
    if prediction[0] == 0:
        return {"prediction": "Non-diabetic"}
    else:
        return {"prediction": "Diabetic"}

# Serve the index.html file
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Mount the static directory to serve the index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())
