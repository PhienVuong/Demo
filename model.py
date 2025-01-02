import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics 


# LOADING DATA
data_df = pd.read_csv("diamonds.csv")
print(data_df.sample(10))

data_df = data_df.drop(["Unnamed: 0"], axis=1)
data_df.shape

# DATA ANALYSIS
# Checking for missing values & categorical variables
# Checking for missing values and categorical variables in the dataset
data_df.info()
# Descriptive Statistics
# Doing Univariate Analysis for statistical description and understanding of dispersion of data
data_df.describe().T
#Doing Bivariate Analysis by examaning a pairplot
# cols = ["#A0522D","#A52A2A","#CD853F","#F4A460","#DEB887"]
# ax = sns.pairplot(data_df, hue= "cut", palette = cols)
# plt.show()

# DATA PREPROCESSING
# Data cleaing
# Removing the feature "Unnamed"
# data_df = data_df.drop(["Unnamed: 0"], axis=1)
# data_df.shape
# Removing the datapoints having min 0 value in either x, y or z features 
data_df = data_df.drop(data_df[data_df["x"]==0].index)
data_df = data_df.drop(data_df[data_df["y"]==0].index)
data_df = data_df.drop(data_df[data_df["z"]==0].index)
data_df.shape
# Removing outliers
# Dropping the outliers (since we have huge dataset) by defining appropriate measures across features 
data_df = data_df[(data_df["depth"]<75)&(data_df["depth"]>45)]
data_df = data_df[(data_df["table"]<80)&(data_df["table"]>40)]
data_df = data_df[(data_df["x"]<40)]
data_df = data_df[(data_df["y"]<40)]
data_df = data_df[(data_df["z"]<40)&(data_df["z"]>2)]
data_df.shape

# Encoding categorical variables
# Making a copy to keep original data in its form intact
data1 = data_df.copy()

# Applying label encoder to columns with categorical data
columns = data1.columns.tolist()
# Example of how encoding was done before training
# Define the model training and encoding process (repeat this as part of saving stage if you need)

# Suppose this was the encoding for the 'cut' column
label_encoder_cut = LabelEncoder().fit(data1['cut'])
label_encoder_color = LabelEncoder().fit(data1['color'])
label_encoder_clarity = LabelEncoder().fit(data1['clarity'])

# Saving encoders
with open('label_encoder_cut.pkl', 'wb') as file:
    pickle.dump(label_encoder_cut, file)

with open('label_encoder_color.pkl', 'wb') as file:
    pickle.dump(label_encoder_color, file)

with open('label_encoder_clarity.pkl', 'wb') as file:
    pickle.dump(label_encoder_clarity, file)
# features = data1.columns.tolist()
expected_columns = [col for col in columns if col != 'price']
with open('dataset_columns.pkl', 'wb') as file:
    pickle.dump(expected_columns, file)
print("Expected columns saved as 'dataset_columns.pkl'")
label_encoder = LabelEncoder()
for col in columns:
    data1[col] = label_encoder.fit_transform(data1[col])
data1.describe()

# Correlation Matrix
# Examining correlation matrix using heatmap
# cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
# cols = (["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
# corrmat= data1.corr()
# f, ax = plt.subplots(figsize=(15,12))
# sns.heatmap(corrmat,cmap=cols,annot=True)
# plt.title('Correlation Heatmap of Diamonds Dataset')
# plt.xticks(rotation=45)
# plt.show()


# MODEL BUILDING
# Defining the independent and dependent variables
X= data1.drop(["price"],axis =1)
y= data1["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=25)
pipeline_xgb=Pipeline([("scalar",StandardScaler()), ("xgb",XGBRegressor(gamma=0, min_child_weight=1, max_depth=6, objective='reg:squarederror', booster='gbtree'))])

pipeline_xgb.fit(X_train, y_train)
# pred = pipeline_xgb.predict(X_test)

# Save the model
with open('pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline_xgb, file)

print("Model saved successfully as 'pipeline.pkl'")