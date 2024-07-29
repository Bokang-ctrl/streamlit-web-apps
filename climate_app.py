import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
#from tensorflow import keras
#from keras.preprocessing.sequence import TimeseriesGenerator
#from tensorflow.keras.losses import MeanSquaredError
#from tensorflow.keras.metrics import RootMeanSquaredError
#from tensorflow.keras.optimizers import Adam

import streamlit as st



st.title("Data Science in Action")

st.write("Hello world... Welcome to my Time Series project 👋")

def load_data():
    return pd.read_csv(r'jena_climate_2009_2016.csv')

data = load_data()
st.write(data.columns)

# Data Preprocessing
data['Date Time'] = pd.to_datetime(data['Date Time'], format = "mixed")
data.set_index('Date Time', inplace = True)
df = data.resample('H').mean()
df1 = df.drop(columns = ['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])


train_df = df1[df1.index < '2014-09-24 18:00:00']
test_df = df1[df1.index > '2016-10-28 11:00:00']               # I can make this the test dataframe

X = test_df.drop(columns = 'T (degC)')
y = test_df['T (degC)']

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)



n_input = 10

def create_sequences(data, n_input):
    sequences = []
    for i in range(len(data) - n_input):
        sequences.append(data[i:i + n_input])
    return np.array(sequences)

test_sequences = create_sequences(X_scaled, n_input)

predictions = model.predict(test_sequences)


