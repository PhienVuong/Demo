import streamlit as st

st.title('Diamonds Price Prediction')

#st.write('Upload your dataset here.')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn. linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

file = st.file_uploader('Choose a CSV file', 'csv')
if file:
    data_df = pd.read_csv(file)
    if "Unnamed: 0" in data_df.columns:
        data_df = data_df.drop(["Unnamed: 0"], axis=1)   

data_df = data_df.drop(data_df[data_df["x"]==0].index)
data_df = data_df.drop(data_df[data_df["y"]==0].index)
data_df = data_df.drop(data_df[data_df["z"]==0].index)

data_df = data_df[(data_df["depth"]<75)&(data_df["depth"]>45)]
data_df = data_df[(data_df["table"]<80)&(data_df["table"]>40)]
data_df = data_df[(data_df["x"]<40)]
data_df = data_df[(data_df["y"]<40)]

data1 = data_df.copy()
data_df = data_df[(data_df["z"]<40)&(data_df["z"]>2)]

columns = ['cut','color','clarity']
label_encoder = LabelEncoder()
for col in columns:
    data1[col] = label_encoder.fit_transform(data1[col])
data1.describe()

cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = (["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
corrmat= data1.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap=cols,annot=True)
plt.title('Correlation Heatmap of Diamonds Dataset')
plt.xticks(rotation=45)

# MODEL BUILDING
# Defining the independent and dependent variables
X= data1.drop(["price"],axis =1)
y= data1["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=25)

# Building pipelins of standard scaler and model for varios regressors.
pipeline_lr=Pipeline([("scalar1",StandardScaler()), ("lr",LinearRegression())])
pipeline_lasso=Pipeline([("scalar2", StandardScaler()), ("lasso",Lasso())])
pipeline_dt=Pipeline([("scalar3",StandardScaler()), ("dt",DecisionTreeRegressor())])
pipeline_rf=Pipeline([("scalar4",StandardScaler()), ("rf",RandomForestRegressor())])
pipeline_kn=Pipeline([("scalar5",StandardScaler()), ("kn",KNeighborsRegressor())])
pipeline_xgb=Pipeline([("scalar6",StandardScaler()), ("xgb",XGBRegressor())])

# List of all the pipelines
pipelines = [pipeline_lr, pipeline_lasso, pipeline_dt, pipeline_rf, pipeline_kn, pipeline_xgb]

# Dictionary of pipelines and model types for ease of reference
pipeline_dict = {0: "LinearRegression", 1: "Lasso", 2: "DecisionTree", 3: "RandomForest",4: "KNeighbors", 5: "XGBRegressor"}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
cv_results_rms = []
for i, model in enumerate(pipelines):
    cv_score = cross_val_score(model, X_train,y_train,scoring="neg_root_mean_squared_error", cv=12)
    cv_results_rms.append(cv_score)

# Model prediction on test data with XGBClassifier which gave us the least RMSE 
pred = pipeline_xgb.predict(X_test)

def Linear(X,y):
    real_price = np.array(X)
    predicted_price = np.array(y)
    sns.scatterplot(x=real_price, y=predicted_price, palette = 'viridis')
    model = LinearRegression()
    model.fit(real_price.reshape(-1,1), predicted_price)
    # plt.scatter(real_price, predicted_price, color='blue')
    predicted_line = model.predict(real_price.reshape(-1,1))
    plt.plot(real_price, predicted_line, color = 'red')

# Draw a scatter chart
Linear(y_test, pred)
plt.xlabel('Real price')
plt.ylabel('Predicted')
plt.title('Compare')
plt.show()
