from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xgboost as xg 
from regressor import NaiveNodeLevelSplitRegressor, SameLevelSplitRegressor, HistogramNodeLevelSplitRegressor

seed = 1988

data = load_diabetes()
dataset = pd.DataFrame(data.data, columns=data.feature_names)
X = dataset.values
Y = data["target"]

test_size = 0.20
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

regressor = HistogramNodeLevelSplitRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("HistogramNodeLevelSplitRegressor: {}".format(np.mean((y_test - y_pred) ** 2)))

regressor = SameLevelSplitRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("SameLevelSplitRegressor: {}".format(np.mean((y_test - y_pred) ** 2)))

regressor = NaiveNodeLevelSplitRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print("NaiveNodeLevelSplitRegressor: {}".format(np.mean((y_test - y_pred) ** 2)))

regressor = xg.XGBRegressor(objective ='reg:squarederror',  n_estimators = 100, seed = seed) 
regressor.fit(X_train, y_train) 
y_pred = regressor.predict(X_test) 
print("XGBRegressor: {}".format(np.mean((y_test - y_pred) ** 2)))
