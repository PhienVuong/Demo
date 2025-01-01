import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (adjust the file path as necessary)
with open('https://github.com/PhienVuong/Demo/blob/master/diamond4.py', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler used during training (if applicable)
# scaler = StandardScaler()
# Note: Uncomment this if you used a scaler during model training

# Define function to make predictions
def predict_price(df):
    # Example: Ensure that columns used in training are present
    features = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']]  # Modify if needed
    
    # Preprocess the features (scaling, encoding, etc.)
    # features_scaled = scaler.transform(features)  # Uncomment if scaling was used
    predictions = model.predict(features)  # Use model.predict(features_scaled) if scaling is used
    df['predicted_price'] = predictions
    return df

# Streamlit UI
st.title("Diamond Price Prediction App")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file with diamond data", type="csv")

if uploaded_file is not None:
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(uploaded_file)

    # Display the first few rows of the uploaded CSV
    st.write("Data preview:")
    st.dataframe(df.head())

    # Check if the required columns are present
    required_columns = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
    if all(col in df.columns for col in required_columns):
        # Predict prices
        df = predict_price(df)

        # Display the predictions
        st.write("Predicted Diamond Prices:")
        st.dataframe(df[['carat', 'cut', 'color', 'clarity', 'predicted_price']].head())
    else:
        st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")

# Add more features or customization if needed
