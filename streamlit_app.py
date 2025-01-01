import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# App title
st.title("Diamonds Price Prediction App 💎")

# File uploader
#file = st.file_uploader("Upload a CSV file containing diamond data", type="csv")

# Load dataset
data_df = pd.read_csv('https://github.com/PhienVuong/Demo/blob/master/diamonds.csv')

# Show uploaded dataset
st.write("Uploaded Dataset Preview:")
st.dataframe(data_df.head())

# Clean the dataset
if "Unnamed: 0" in data_df.columns:
data_df = data_df.drop(["Unnamed: 0"], axis=1)

    data_df = data_df.drop(data_df[data_df["x"] == 0].index)
    data_df = data_df.drop(data_df[data_df["y"] == 0].index)
    data_df = data_df.drop(data_df[data_df["z"] == 0].index)
    data_df = data_df[(data_df["depth"] < 75) & (data_df["depth"] > 45)]
    data_df = data_df[(data_df["table"] < 80) & (data_df["table"] > 40)]
    data_df = data_df[(data_df["x"] < 40)]
    data_df = data_df[(data_df["y"] < 40)]
    data_df = data_df[(data_df["z"] < 40) & (data_df["z"] > 2)]

    st.write("Cleaned Dataset Preview:")
    st.dataframe(data_df.head())

    # Process dataset
    data1 = data_df.copy()

    # Encode categorical columns
    categorical_columns = ["cut", "color", "clarity"]
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        if col in data1.columns:
            data1[col] = label_encoder.fit_transform(data1[col])
        else:
            st.warning(f"Column '{col}' not found in the dataset. Skipping encoding for this column.")

    # Check if 'price' column exists
    if "price" not in data1.columns:
        st.error("The dataset must contain a 'price' column for training the model.")
    else:
        # Define independent (X) and dependent (y) variables
        X = data1.drop(["price"], axis=1)
        y = data1["price"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Build a model pipeline
        pipeline_rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor(random_state=42))])

        # Train the model
        pipeline_rf.fit(X_train, y_train)

        # Predict prices for the entire dataset
        data1["predicted_price"] = pipeline_rf.predict(X)

        # Show the table with predicted prices
        st.subheader("Dataset with Predicted Prices:")
        st.dataframe(data1[["price", "predicted_price"]].head())

        # Allow users to download the table
        csv = data1.to_csv(index=False)
        st.download_button(
            label="Download Predicted Prices as CSV",
            data=csv,
            file_name="predicted_prices.csv",
            mime="text/csv"
        )
else:
    st.info("Please upload a CSV file to begin.")

# File uploader
file = st.file_uploader("Upload a CSV file containing diamond data", type="csv")
