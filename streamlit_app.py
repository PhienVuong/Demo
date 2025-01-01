import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the trained machine learning model (make sure the model is saved as a .pkl file)
model = joblib.load('diamond_price_model.pkl')

# If your model requires scaling (like using StandardScaler), load it
scaler = joblib.load('scaler.pkl')  # Optional, depends on your model's requirements

def predict_price(df):
    """
    This function takes a dataframe and returns the predicted price for each diamond.
    It assumes the model requires preprocessed data (e.g., scaling).
    """
    # If the model requires scaling, scale the features accordingly
    if scaler:
        df_scaled = scaler.transform(df)
    else:
        df_scaled = df  # If no scaling is needed
    
    # Make predictions using the model
    predictions = model.predict(df_scaled)
    return predictions

# Streamlit App Interface
def main():
    st.title('Diamond Price Prediction App')
    st.write("Upload a CSV file containing the diamond attributes (like carat, cut, color, clarity, etc.) and get predicted prices.")

    # File uploader widget to upload a CSV
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded CSV file into a pandas DataFrame
            df = pd.read_csv(uploaded_file)

            # Show a preview of the data
            st.write("Preview of your uploaded data:")
            st.write(df.head())

            # Check if the required columns exist in the DataFrame
            required_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
            if all(col in df.columns for col in required_columns):
                # Handle missing values if necessary (you can improve this based on your model's needs)
                if df.isnull().sum().any():
                    st.warning("Your data contains missing values. They will be handled with imputation or removed if necessary.")

                # Predict prices using the model
                predictions = predict_price(df[required_columns])

                # Add the predictions as a new column in the dataframe
                df['Predicted Price'] = predictions

                # Display the dataframe with the predicted prices
                st.write("Predicted Diamond Prices:")
                st.write(df)

            else:
                st.error(f"The uploaded file is missing required columns. Expected columns: {required_columns}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.info("Please upload a CSV file to predict diamond prices.")

if __name__ == '__main__':
    main()
