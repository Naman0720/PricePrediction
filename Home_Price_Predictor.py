#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Data.csv')

# Preprocess the data
# Handle missing values and outliers if necessary
# Perform feature engineering if required
# Check for missing values
# Handle missing values
data = data.dropna()


# Split the data into training and testing sets
X = data[['UNEMP_RATE','TOT_POP','CONS_SENT','UND_CONST']]
y = data['HOM_PRC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Interpret the model coefficients or feature importance scores
coefficients = model.coef_
feature_importance = pd.Series(coefficients, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)

# Use the trained model to make predictions on new data if needed
# new_data = pd.read_csv('new_data.csv')
# new_predictions = model.predict(new_data)


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Data.csv')

# Preprocess the data
# Handle missing values and outliers if necessary
# Perform feature engineering if required
# Check for missing values
# Handle missing values
data = data.dropna()

# Split the data into training and testing sets
X = data[['UNEMP_RATE', 'TOT_POP', 'CONS_SENT', 'UND_CONST']]
y = data['HOM_PRC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Interpret the model feature importance scores
feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importance:")
print(feature_importance)

# Use the trained model to make predictions on new data if needed
# new_data = pd.read_csv('new_data.csv')
# new_predictions = model.predict(new_data)

import matplotlib.pyplot as plt

# Get feature importance
feature_importance = model.feature_importances_

# Create a bar plot of feature importance
plt.figure(figsize=(8, 6))
plt.bar(X.columns, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance - Random Forest')
plt.xticks(rotation=45)
plt.show()

# Convert the 'DATE' column to datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# Extract the year from the 'DATE' column
data['Year'] = data['DATE'].dt.year

# Group the data by year and calculate the average home price for each year
average_prices = data.groupby('Year')['HOM_PRC'].mean()

# Create a line plot to show the trend of home prices over the years
plt.figure(figsize=(10, 6))
plt.plot(average_prices.index, average_prices.values)
plt.xlabel('Year')
plt.ylabel('Average Home Price')
plt.title('Home Prices Over the Last 20 Years')
plt.xticks(average_prices.index, rotation=45)
plt.show()


# In[8]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Data.csv')

# Preprocess the data
# Handle missing values and outliers if necessary
# Perform feature engineering if required
# Check for missing values
# Handle missing values
data = data.dropna()

# Split the data into training and testing sets
X = data[['UNEMP_RATE', 'TOT_POP', 'CONS_SENT', 'UND_CONST']]
y = data['HOM_PRC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = SVR()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Feature Importance:")
print(feature_importance)

# Use the trained model to make predictions on new data if needed
# new_data = pd.read_csv('new_data.csv')
# new_predictions = model.predict(new_data)


# In[9]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Data.csv')

# Preprocess the data
# Handle missing values and outliers if necessary
# Perform feature engineering if required
# Check for missing values
# Handle missing values
data = data.dropna()

# Split the data into training and testing sets
X = data[['UNEMP_RATE', 'TOT_POP', 'CONS_SENT', 'UND_CONST']]
y = data['HOM_PRC']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)
print("Feature Importance:")
print(feature_importance)
# Use the trained model to make predictions on new data if needed
# new_data = pd.read_csv('new_data.csv')
# new_predictions = model.predict(new_data)


# In[ ]:




