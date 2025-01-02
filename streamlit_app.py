import streamlit as st
import streamlit as st
import pandas as pd
import pickle
import numpy as np
# Load the pre-trained model and expected columns
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('dataset_columns.pkl', 'rb') as file:
    expected_columns = pickle.load(file)
# Load the pre-trained model and expected columns
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

with open('dataset_columns.pkl', 'rb') as file:
    expected_columns = pickle.load(file)

# Load encoders
with open('label_encoder_cut.pkl', 'rb') as file:
    label_encoder_cut = pickle.load(file)

with open('label_encoder_color.pkl', 'rb') as file:
    label_encoder_color = pickle.load(file)

with open('label_encoder_clarity.pkl', 'rb') as file:
    label_encoder_clarity = pickle.load(file)
# Function to preprocess the input data and make predictions
def preprocess_and_predict(new_data_file):
    new_data = pd.read_csv(new_data_file)
    missing_columns = [col for col in expected_columns if col not in new_data.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return None
     # Encode categorical columns
    new_data['cut'] = label_encoder_cut.transform(new_data['cut'])
    new_data['color'] = label_encoder_color.transform(new_data['color'])
    new_data['clarity'] = label_encoder_clarity.transform(new_data['clarity'])
    
    # Ensure columns are in the correct order
    new_data = new_data[expected_columns]  
    new_data = new_data[expected_columns]  # Ensure the correct columns
    predictions = pipeline.predict(new_data)
    predictions = np.absolute(pipeline.predict(new_data)) 
    
    new_data['predictions'] = predictions
    return new_data

# Streamlit app interface
st.title("Simple Diamond Prediction")

uploaded_file = st.file_uploader("Upload your data file", type=["csv"])

if uploaded_file is not None:
    new_data = preprocess_and_predict(uploaded_file)
    if new_data is not None:
        st.write("Predictions:")
        st.write(new_data)
        st.download_button("Download Predictions", new_data.to_csv(index=False).encode('utf-8'), file_name="predictions.csv", mime='text/csv')
