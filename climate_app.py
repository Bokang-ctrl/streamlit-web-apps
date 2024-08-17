import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


if 'data_loaded' not in st.session_state:
    st.write("Initializing the app, please wait while data is being loaded... can take up to a minute.")



@st.cache_resource
def load_datasets():
    data = pd.read_csv('jena_climate_2009_2016.csv')
    train_preds = pd.read_csv('train_predictions.csv')
    test_preds = pd.read_csv('test_predictions.csv')
    val_preds = pd.read_csv('validation_predictions.csv')
    
    # Set the date column as index
    train_preds.set_index('Date', inplace=True)
    train_preds.index = pd.to_datetime(train_preds.index)
    test_preds.set_index('Date', inplace=True)
    test_preds.index = pd.to_datetime(test_preds.index)
    val_preds.set_index('Date', inplace=True)
    val_preds.index = pd.to_datetime(val_preds.index)
    
    return data, train_preds, test_preds, val_preds

#

# Load datasets and store them in session state if not already present
if 'data' not in st.session_state:
    data, train_preds, test_preds, val_preds = load_datasets()
    st.session_state['data'] = data
    st.session_state['train_preds'] = train_preds
    st.session_state['test_preds'] = test_preds
    st.session_state['val_preds'] = val_preds
    st.session_state['data_loaded'] = True  # Mark data as loaded
else:
    data = st.session_state['data']
    train_preds = st.session_state['train_preds']
    test_preds = st.session_state['test_preds']
    val_preds = st.session_state['val_preds']

# Clear the loading message once data is loaded
if 'data_loaded' in st.session_state:
    st.empty()  # This clears the previous message





# Data Preprocessing
if 'df2' not in st.session_state:
    df = data.copy()
    df['Date Time'] = pd.to_datetime(df['Date Time'], format="mixed")
    df.set_index('Date Time', inplace=True)
    df1 = df.resample('h').mean()
    df2 = df1.drop(columns=['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])

    st.session_state['df2'] = df2
else:
    df2 = st.session_state['df2']


# Splitting the data
if 'temps_of_train' not in st.session_state:
    train_val_temps = df2.loc[df2.index < '2014-09-24 18:00:00']['T (degC)']

    # Train
    train_len = round(len(train_val_temps) * 0.7)
    temps_of_train = train_val_temps[10:train_len]

    # Test
    temps_of_test = df2[df2.index > '2016-10-28 11:00:00']['T (degC)'][10:]

    # Validation
    val_len = round(len(train_val_temps) * 0.3)
    temps_of_validation = train_val_temps[-val_len + 10:]

    st.session_state['temps_of_train'] = temps_of_train
    st.session_state['temps_of_test'] = temps_of_test
    st.session_state['temps_of_validation'] = temps_of_validation
else:
    temps_of_train = st.session_state['temps_of_train']
    temps_of_test = st.session_state['temps_of_test']
    temps_of_validation = st.session_state['temps_of_validation']



# Evaluation function
def get_mse(y_true, preds):
    return round(mean_squared_error(y_true, preds), 3)


# -------------------  Web Page
st.title("Data Science in Action \n")


# Sidebar and Plot Logic
st.sidebar.title("Filters")

predictions_to_show = st.sidebar.selectbox('Predictions sample', ["None", "Validation", "Test"])
quarter_to_plot = st.sidebar.selectbox('Quarter', ['All', 'First', 'Second', 'Third', 'Fourth'])




if predictions_to_show == 'None':
    st.write("#### Hello world...    👋")

    st.write("Welcome to my Time Series Analysis webpage. Here I will demonstrate temperature forecasting using the LSTM (Long Short-Term Memory) model, which is an example of a Recurrent Neural Network. The dataset used is the Jena Climate dataset provided by Kaggle.")

    st.write("\nBelow, you can see the structure of the dataset. It will be split into training, validation, and test samples. The LSTM model will then be trained on the data to make predictions.")


    st.write(data.head())
    st.write(f'The shape of the data is {data.shape}')
    st.write('\n')


    landing_page_T_plot = df2.resample('d').mean()

    from Plot_functions import all_samples_plot_quarter
    all_samples_plot_quarter(quarter_to_plot, landing_page_T_plot)





elif predictions_to_show == 'Validation':
    from Plot_functions import validation_plot_quarter_with_year, validation_plot_quarter_no_year

    st.write("Now that we have trained our model, we will make predictions on the validation sample and evaluate them.")


    st.write("The Mean Squared Error of the Validation predictions is ", get_mse(temps_of_validation, val_preds['Predicted T (degC)']))

    mape_val = round(mean_absolute_percentage_error(temps_of_validation, val_preds['Predicted T (degC)']), 2)
    st.write(f"This is a good model with a mean absolute percentage error of {mape_val}%")

    val_plot_df = pd.DataFrame({'True Temperature': temps_of_validation, 'Predicted Temperature': val_preds['Predicted T (degC)']})
    val_plot_df['Date'] = pd.to_datetime(val_plot_df.index, format='mixed')

    val_year = st.sidebar.selectbox('Year', ["All", "2013", "2014"])

    if val_year != "All":
        val_plot_df_1 = val_plot_df[val_plot_df.Date.dt.year == int(val_year)]
        validation_plot_quarter_with_year(quarter_to_plot, val_plot_df_1, val_year)
    else:
        validation_plot_quarter_no_year(quarter_to_plot, val_plot_df)





else:
    st.write("Now to see our model performance on data it has not seen, we will evaluate the Test predictions against true Test values.")
    st.write("The Mean Squared Error of the Test predictions is  ", get_mse(temps_of_test, test_preds['Predicted T (degC)']))

    mape_test = round(mean_absolute_percentage_error(temps_of_test, test_preds['Predicted T (degC)']), 2)
    st.write(f"The model performs good with a mean absolute percentage error of {mape_test}%.")

    test_plot_df = pd.DataFrame({'True Temperature': temps_of_test, 'Predicted Temperature': test_preds['Predicted T (degC)'], 'Date' : temps_of_test.index})

    from Plot_functions import test_plot_quarter_no_year

    test_plot_quarter_no_year(quarter_to_plot, test_plot_df)
