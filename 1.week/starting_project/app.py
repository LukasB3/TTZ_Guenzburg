import streamlit
import pandas as pd
import requests

df = pd.read_csv("/Users/lukasbossert/Documents/VS-Code/1TTZ_first/starting_project/data/cleaned_car_data.csv")
print(df.head)

data_list = []

def run():
    streamlit.title("Car Price Prediction")
    name = streamlit.selectbox("Cars Model", df.name.unique())
    company = streamlit.selectbox("Company Name", df.company.unique())
    year = streamlit.number_input("Year")
    kms_driven = streamlit.number_input("Kilometers driven")
    fuel_type = streamlit.selectbox("Fuel type", df.fuel_type.unique())

    data = {
    'name': name,
    'company': company,
    'year': year,
    'kms_driven': kms_driven,
    'fuel_type': fuel_type,
    }

    if streamlit.button("Predict"):
        response = requests.post("http://127.0.0.1:8000/predict", json=data)
        prediction = response.text
        streamlit.success(f"The prediction from model: {prediction}")


if __name__ == '__main__':
    run()

