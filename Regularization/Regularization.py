

# Source: https://www.youtube.com/watch?v=VqKq78PVO9g&ab_channel=codebasics

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset = pd.read_csv('./Melbourne_housing_FULL.csv')
print(dataset.shape)
print(dataset.isna().sum())

# cleaning up data:
#  ================
cols_to_use = ['Suburb', 'Rooms', 'Type', 'Method', 'SellerG', 'Regionname', 'Propertycount', 'Distance',
               'CouncilArea', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 'BuildingArea', 'Price']
dataset = dataset[cols_to_use]
print(dataset.head())

print(dataset.isna().sum())
cols_to_fill_zero = ['Propertycount', 'Distance', 'Bedroom2', 'Bathroom', 'Car']
dataset[cols_to_fill_zero] = dataset[cols_to_fill_zero].fillna(0)

dataset['Landsize'] = dataset['Landsize'].fillna(dataset.Landsize.mean())
dataset['BuildingArea'] = dataset['BuildingArea'].fillna(dataset.BuildingArea.mean())

dataset.dropna(inplace=True)
print(dataset.isna().sum())
#  hot encoding: https://www.youtube.com/watch?v=9yl6-HEY7_s&ab_channel=codebasics
dataset = pd.get_dummies(dataset, drop_first=True)
#  ================

x = dataset.drop('Price', axis=1)
y = dataset['Price']

from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_x, train_y)
accuracy = reg.score(train_x, train_y)
print('error on train (bias):', 1-accuracy)
accuracy = reg.score(test_x, test_y)
print('error on test (variance):', 1-accuracy)
print('the linear regression model is OVERFITTED!')

# Use sklearn Lasso regression (L1 regurarization)
from sklearn import linear_model
lasso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(train_x, train_y)
print(50*'=')
print('Lasso Regression')
accuracy = lasso_reg.score(train_x, train_y)
print('error on train (bias):', 1-accuracy)
accuracy = lasso_reg.score(test_x, test_y)
print('error on test (variance):', 1-accuracy)
print('the lasso regression model is FITTED!')

# Use sklearn Ridge regression (L2 regurarization)
ridge_reg = linear_model.Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(train_x, train_y)
print(50*'=')
print('Ridge Regression')
accuracy = ridge_reg.score(train_x, train_y)
print('error on train (bias):', 1-accuracy)
accuracy = ridge_reg.score(test_x, test_y)
print('error on test (variance):', 1-accuracy)
print('the ridge regression model is FITTED!')

