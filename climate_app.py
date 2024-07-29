import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import streamlit as st



st.title("Data Science in Action")

st.write("Hello world... Welcome to my Time Series project 👋")

def load_data():
    return pd.read_csv(r'jena_climate_2009_2016.csv')

data = load_data()
st.write(data.columns)
