import streamlit as st

st.title('Diamonds Price Prediction')

#st.write('Upload your dataset here.')
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

# App title
st.title("Diamonds Price Prediction")

# File uploader
file = st.file_uploader("Choose a CSV file", type="csv")

if file:
    # Load dataset
    data_df = pd.read_csv(file)

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

    st.write("Dataset after cleaning:")
    st.write(data_df.describe())

    # Copy dataset for further processing
    data1 = data_df.copy()

    # Encode categorical columns
    columns = ["cut", "color", "clarity"]
    label_encoder = LabelEncoder()
    for col in columns:
        if col in data1.columns:
            data1[col] = label_encoder.fit_transform(data1[col])
        else:
            st.warning(f"Column '{col}' not found in the dataset. Skipping encoding for this column.")

    st.write("Dataset after preprocessing:")
    st.write(data1.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corrmat = data1.corr()
    plt.figure(figsize=(15, 12))
    sns.heatmap(corrmat, cmap="coolwarm", annot=True)
    st.pyplot(plt)

    # Splitting the dataset
    X = data1.drop(["price"], axis=1)
    y = data1["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=25)

    # Model pipelines
    pipeline_lr = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    pipeline_lasso = Pipeline([("scaler", StandardScaler()), ("lasso", Lasso())])
    pipeline_dt = Pipeline([("scaler", StandardScaler()), ("dt", DecisionTreeRegressor())])
    pipeline_rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestRegressor())])
    pipeline_kn = Pipeline([("scaler", StandardScaler()), ("kn", KNeighborsRegressor())])
    pipeline_xgb = Pipeline([("scaler", StandardScaler()), ("xgb", XGBRegressor())])

    pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]
    pipeline_dict = {
        0: "Linear Regression",
        1: "Lasso",
        2: "Decision Tree",
        3: "Random Forest",
        4: "K-Neighbors",
        5: "XGBoost",
    }

    # Train models and evaluate
    st.subheader("Model Training")
    for i, pipe in enumerate(pipelines):
        pipe.fit(X_train, y_train)
        score = cross_val_score(pipe, X_train, y_train, scoring="neg_root_mean_squared_error", cv=5)
        st.write(f"{pipeline_dict[i]}: Mean RMSE = {-np.mean(score):.2f}")

    # Prediction using XGBoost
    pred = pipeline_xgb.predict(X_test)

    # Visualization
    st.subheader("Prediction Results")
    fig, ax = plt.subplots()
    sns.scatterplot(x=y_test, y=pred, ax=ax)
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted Prices")
    st.pyplot(fig)
else:
    st.warning("Please upload a CSV file.")

