
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("iris_model.pkl")

# Application title
st.title("Iris Flower Classification")

st.write("Enter flower measurements below:")

# Input fields
sl = st.number_input("Sepal Length", value=5.1)
sw = st.number_input("Sepal Width", value=3.5)
pl = st.number_input("Petal Length", value=1.4)
pw = st.number_input("Petal Width", value=0.2)

# Prediction
if st.button("Predict Species"):

    input_data = pd.DataFrame([[
        sl, sw, pl, pw
    ]], columns=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ])

    result = model.predict(input_data)

    st.success("Predicted Species: " + result[0])
