# app.py
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("linear_reg_best.pkl")
selector = joblib.load("feature_selector.pkl")

st.title("🏠 House Price Estimator")

# Take input values
input_data = {}
for col in selected_features:
    input_data[col] = st.number_input(col)

input_df = pd.DataFrame([input_data])
X_input = selector.transform(input_df)
prediction = model.predict(X_input)

st.subheader("Predicted Price:")
st.write(f"${prediction[0]:,.2f}")
