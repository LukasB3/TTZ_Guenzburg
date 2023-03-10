from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import uvicorn
import streamlit


app = FastAPI(title='Car Price Prediction', version='1.0',
description='Linear Regression model is used for prediction')

model = joblib.load("/Users/lukasbossert/Documents/VS-Code/1TTZ_first/starting_project/LinearRegressionModel.joblib")

class Data(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: float
    fuel_type: str


@app.get('/')
@app.get('/home')
def read_home():
    """
    Home endpoint which can be used to test the availability of the    application.
    """
    return {'message': 'System is healthy'}

@app.post("/predict")
def predict(data: Data):
    result = model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([data.name,data.company,data.year,data.kms_driven,data.fuel_type]).reshape(1,5)))[0]
    return result

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)