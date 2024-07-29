import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import streamlit as st



# dataset names
df1 = 'jena_climate_2009_2016.csv'
train = 'train_predictions.csv'
test = 'test_predictions.csv'
validation = 'validation_predictions.csv'

# Load datasets

data = pd.read_csv(df1)
train_preds = pd.read_csv(train)
test_preds = pd.read_csv(test)
val_preds = pd.read_csv(validation)



# Data Preprocessing

df = data.copy()
df['Date Time'] = pd.to_datetime(df['Date Time'], format = "mixed")
df.set_index('Date Time', inplace = True)
df1 = data.resample('H').mean()
df2 = df1.drop(columns = ['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])


# What I need to do is to take df2 target (T degC) and split it into Train Test and Validation splits so that I can run mean squared error and perfrom evaluation
y_true_1 = df2.loc[df2.index < '2014-09-24 18:00:00']['T (degC)']
y_true_test = df2[df2.index > '2016-10-28 11:00:00']['T (degC)']                # Test Target is done and ready

val_len = round(len(df2) * 0.3)

y_true_train = y_true_1[:-val_len]
y_true_validation = y_true_1[-val_len:]


# -------------------  Web Page
st.title("Data Science in Action \n")

st.write("#### Hello world👋")

st.write("Welcome to my Time Series Analysis project. Here I demonstrate temperature forecasting with LSTM. An example of a Recurrent Neural Network.")

st.write("\n Below is how the climate dataset looks like.")



st.write(data.head())
st.write(train_preds.head())
st.write(test_preds.head())
st.write(val_preds.head())


 
 # Evaluation metric 
def get_mse(y_true, preds):
    return mean_squared_error(y_true, preds)

# Here I will test if the function works well
st.write(get_mse(y_true_train, train_preds['Predicted T (degC)']))

