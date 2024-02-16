## Dragon Real Estate- Price Predicted
import pandas as pd

housing = pd.read_csv("data.csv")
housing.head()
housing.info()
housing['CHAS'].value_counts()
housing.describe()
# %matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
## Train-Test Spliting 
import numpy as np
# from sklearn.model_selection import train_test_split
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
 
# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in Train set: {len(train_set)}")
# print(f"Rows in Test set: {len(test_set)}")
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing,test_size = 0.2, random_state = 42 )
print(f"Rows in Train set: {len(train_set)}")
print(f"Rows in Test set: {len(test_set)}")
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state=42)
for train_index , test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


strat_test_set['CHAS'].value_counts()
strat_train_set['CHAS'].value_counts()
# 95/7
# 376/28
housing = strat_train_set.copy()
## Looking for Correlations
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=1.0)
## Trying out Attribute Combination
housing["TAXRM"] = housing["TAX"] / housing["RM"]
housing.head()
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)
housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=1.0)
housing["RM"].value_counts()
housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()
## Missing Attributes
housing["RM"].info()
housing.info()
# to remove the rows of whole data where the value of RM is null or empty(Option-1)
a = housing.dropna(subset=['RM'])
a.shape
# note that the original hudsing dataframe remain unchange
median = housing['RM'].median()
housing['RM'].fillna(median)
housing.describe()
housing.shape
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
imputer.statistics_.shape
imputer.statistics_
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns = housing.columns)
housing_tr.describe()
## Scikit-Learn Design
 
## Creating Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ("std_scaler", StandardScaler()),
])
housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr.shape
## Selecting the desired Model
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5] 
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

rmse
## Cross validation Technique
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores) 
rmse_scores
def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())
print_scores(rmse_scores)
## Saving he Model
from joblib import dump, load
dump(model, 'MyDragon.joblib')
## Testing the Model on Test data
X_test = strat_test_set.drop('MEDV', axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions)
print(list(Y_test))
final_rmse
prepared_data[0]
## Using the model
from joblib import dump, load
import numpy as np
model = load('MyDragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.23979304, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  0.41164221, -0.86091034]])
model.predict(features)
