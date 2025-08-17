import numpy as np
import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

st.title('Electricity Cost Prediction')

# Site Area
site_area = st.slider("Site Area", min_value=500.0, step=5.0, max_value=5000.0)

# Structure type
structure_type=st.selectbox('Select structure type', options=['Residential', 'Commercial', 'Mixed-use', 'Industrial'])

with open("cost_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Water Consumption
water_consumption = st.slider("Water Consumption", min_value=1000.0, max_value=11000.0, step=10.5)

# Recycling Rate
recycling_rate = st.slider("Recycling Rate", min_value=10, max_value=100, step=1)

# Utilisation Rate
utilisation_rate = st.slider("Utilisation Rate", min_value=10, max_value=100, step=1)

# Air Quality Index
air_quality_index = st.slider("Air Quality Index", min_value=0, max_value=200, step=1)

# Issue Resolution Time
issue_resolution_time = st.slider("Issue Resolution Time", min_value=1, max_value=80, step=1)

# Resident Count
resident_count = st.slider("Resident Count", min_value=0, max_value=500, step=1)

# Make DataFrame for model input
input_df = pd.DataFrame([{
    "site area": site_area,
    "water consumption": water_consumption,
    "recycling rate": recycling_rate,
    "utilisation rate": utilisation_rate,
    "air quality index": air_quality_index,
    "issue resolution time": issue_resolution_time,
    "resident count": resident_count,
    "structure type": structure_type
}])
# Encode categorical column using the same encoder as training
encoded_array = encoder.transform(input_df[["structure type"]])
encoded_cols = encoder.get_feature_names_out(["structure type"])
encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=input_df.index)

# Drop original categorical column & concatenate encoded features
input_final = pd.concat([input_df.drop(columns=["structure type"]), encoded_df], axis=1)

# Separate numeric and categorical parts
numeric_features = input_final.iloc[:, :7].values   # first 7 columns
categorical_features = input_final.iloc[:, 7:].values  # remaining columns

# Scale numeric features
with open("cost_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

scaled_features = scaler.transform(numeric_features)

# Combine back
final_input = np.hstack((scaled_features, categorical_features))

if st.button('Predict electricity cost'):
    with open('Electricity_cost_prediction.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(final_input)
    st.success(f"Predicted Electricity Cost: {prediction[0]}")