#!/usr/bin/env python
# coding: utf-8

# ## Importing the Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime


# ## Loading the Dataset

# In[9]:


data = pd.read_csv(r"C:\Users\yashw\OneDrive\Desktop\Tcs merge data.csv")
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values(by='Date',inplace=True)
data.head()


# ## Data Preprocessing 

# In[10]:


#checking for null values
print(data.isnull().sum())


# In[11]:


#Converting numerical columns
data['Open'] = pd.to_numeric(data['Open'], errors='coerce')
data['High'] = pd.to_numeric(data['High'], errors='coerce')
data['Low'] = pd.to_numeric(data['Low'], errors='coerce')
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')


# In[14]:


#Fill any remaining NaN Values
data.fillna(method= 'ffill', inplace=True)


# ## Exploratory Data Analysis

# In[18]:


#ploting close price over time

plt.figure(figsize=(12,6))
plt.plot(data['Date'],data['Close'], color='blue',
label='Close price')
plt.xlabel('Date')
plt.ylabel('Stocl price')
plt.title('TCS Stock Close Price Over Time')
plt.legend()
plt.show()


# In[21]:


#Calculating 50-day and 200-day moviing averages

data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()


# In[28]:


#Plot with moving Averages

plt.figure(figsize=(12,6))
plt.plot(data['Date'], data['Close'], label='Close price')
plt.plot(data['Date'], data['MA50'], label='50-Day MA')
plt.plot(data['Date'], data['MA200'], label='200-Day MA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('TCS Stock Price with Moving Averages')
plt.legend()
plt.show()


# ## Feature Engineering

# In[44]:


#Extracting  features like Year, Month, Day, Day of Week from Date.
#Creating  lag features (e.g., previous day’s close, previous day’s high/low).

data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day
data['Day_of_Week'] = data['Date'].dt.dayofweek


# In[45]:


#Lag features

data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)     #Drop rows NaN values from Shifting


# ## Model Building and Prediction

# ### Using Linear Regression to predict the Close price based on features.
# 
# 

# ### Train/Test Split for model evaluation.

# In[46]:


# Feature Selection
X = data[['Open', 'High', 'Low', 'Volume', 'Prev_Close', 'Day_of_Week', 'Month']]
y = data['Close']

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)

#Linear regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test,y_pred))
print("R-Squared Score:",    r2_score(y_test,y_pred))


# ## Visualizing model Performance

# In[47]:


## Plot predicted vs. actual values.
## Scatter plot to observe prediction accuracy.

plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, color='blue', label='Prediction vs Actual')
plt.xlabel('Actual close Price')
plt.ylabel('Predicted Close price')
plt.title('Actual vs Predicted Close Price')
plt.legend()
plt.show()


# In[ ]:




