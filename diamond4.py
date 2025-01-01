# IMPORT LIBRARIES
import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

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

# Note: 
# There are 53940 non-null values in all the attributes thus no missing values.
# Datatype of features 'cut', 'color' & 'clarity' is "object" which needs to be converted into numerical variable (will be done in data preprocessing) before we feed the data to algorithms



# Evaluating categorical features
plt.figure(figsize=(10,8))
cols = ["#A0522D","#A52A2A","#CD853F","#F4A460","#DEB887"]
ax = sns.violinplot(x="cut",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Cut for Price", color="#774571", fontsize = 20)
ax.set_ylabel("Price", color="#4e4c39", fontsize = 15)
ax.set_xlabel("Cut", color="#4e4c39", fontsize = 15)
plt.show()

plt.figure(figsize=(12,8))
ax = sns.violinplot(x="color",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Colors for Price", color="#774571", fontsize = 20)
ax.set_ylabel("Price", color="#4e4c39", fontsize = 15)
ax.set_xlabel("Color", color="#4e4c39", fontsize = 15)
plt.show()

plt.figure(figsize=(13,8))
ax = sns.violinplot(x="clarity",y="price", data=data_df, palette=cols,scale= "count")
ax.set_title("Diamond Clarity for Price", color="#774571", fontsize = 20)
ax.set_ylabel("Price", color="#4e4c39", fontsize = 15)
ax.set_xlabel("Clarity", color="#4e4c39", fontsize = 15)
plt.show()


# Note: 
# "Ideal" diamond cuts are the most in the number while the "Fair" is the least. More diamonds of all of such cuts for lower price category.
# "J" color diamond which is worst are most rare however, "H" and "G" are more in number eventhough they're of inferior quality as well.
# Diamonds of "IF" clarity which is best as well as "I1" which is worst are very rare and rest are mostly of in-between clarities


# Descriptive Statistics
# Doing Univariate Analysis for statistical description and understanding of dispersion of data
data_df.describe().T
'''
# Note: 
"Price" as expected is right skewed, having more number of data points in left
Under dimensional features of 'x', 'y' & 'z' - min value is 0 thus making such datapoints 
either a 1D or 2D diamond object which doesn't make much sense - so needs either to be imputed with appropriate value or dropped altogether
'''

#Doing Bivariate Analysis by examaning a pairplot
ax = sns.pairplot(data_df, hue= "cut", palette = cols)
plt.show()

'''
Note: ¶
There's a useless feature "unnamed" which is just an index and needs to be eliminated altogether.
Features are having datapoints that are far from the rest of the dataset (outliers) which needs to be dealth with or else would affect our model.
"y" and "z" have some dimensional outliers in our dataset that needs to be eliminated.
Features "depth" & "table" should be capped after we confirm by examining the Line plots.
'''


# Checking for Potential Outliers
lm = sns.lmplot(x="price", y="y", data=data_df, scatter_kws={"color": "#BC8F8F"}, line_kws={"color": "#8B4513"})
plt.title("Line Plot on Price vs 'y'", color="#774571", fontsize = 10)
plt.show()

lm = sns.lmplot(x="price", y="z", data=data_df, scatter_kws={"color": "#BC8F8F"}, line_kws={"color": "#8B4513"})
plt.title("Line Plot on Price vs 'z'", color="#774571", fontsize = 10)
plt.show()

lm = sns.lmplot(x="price", y="depth", data=data_df, scatter_kws={"color": "#BC8F8F"}, line_kws={"color": "#8B4513"})
plt.title("Line Plot on Price vs 'depth'", color="#774571", fontsize = 10)
plt.show()

lm = sns.lmplot(x="price", y="table", data=data_df, scatter_kws={"color": "#BC8F8F"}, line_kws={"color": "#8B4513"})
plt.title("Line Plot on Price vs 'Table'", color="#774571", fontsize = 10)
plt.show()

'''
Note: 
In the Line plots of above features, we can easily spot the outliers which we'll drop before feeding the data to the algorithm
'''

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
columns = ['cut','color','clarity']
label_encoder = LabelEncoder()
for col in columns:
    data1[col] = label_encoder.fit_transform(data1[col])
data1.describe()

'''
Note: 
As categorical features have been converted into numerical columns, we are getting 5-point summary along with count, mean & std for them as well.
Now, we may analyze correlation matrix after getting done with pre-processing for possible feature selection in order to make our dataset more cleaner,
optimal before we feed it into algorithm
'''

# Correlation Matrix
# Examining correlation matrix using heatmap
cmap = sns.diverging_palette(205, 133, 63, as_cmap=True)
cols = (["#682F2F", "#9E726F", "#D6B2B1", "#B9C0C9", "#9F8A78", "#F3AB60"])
corrmat= data1.corr()
f, ax = plt.subplots(figsize=(15,12))
sns.heatmap(corrmat,cmap=cols,annot=True)
plt.title('Correlation Heatmap of Diamonds Dataset')
plt.xticks(rotation=45)
plt.show()

'''
Note:
Features "carat", "x", "y", "z" are highly correlated to our target variable, price.
Features "cut", "clarity", "depth" are very low correlated (<|0.1|) thus may be removed 
though due to presence of only few selected features, we won't be doing that.
'''


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
    print("%s: %f " % (pipeline_dict[i], -1 * cv_score.mean()))
    
# Model prediction on test data with XGBClassifier which gave us the least RMSE 
pred = pipeline_xgb.predict(X_test)
print("R^2:",metrics.r2_score(y_test, pred))
print("Adjusted R^2:",1 - (1-metrics.r2_score(y_test, pred))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))
print("MAE:",metrics.mean_absolute_error(y_test, pred))
print("MSE:",metrics.mean_squared_error(y_test, pred))
print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, pred)))

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