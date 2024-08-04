import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

import streamlit as st
import plotly.graph_objects as go



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
df1 = df.resample('h').mean()
df2 = df1.drop(columns = ['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])


# set the date column as index
train_preds.set_index('Date', inplace = True)
test_preds.set_index('Date', inplace = True)
val_preds.set_index('Date', inplace = True)



# What I need to do is to take df2 target (T degC) and split it into Train Test and Validation splits so that I can run mean squared error and perfrom evaluation
train_val_temps = df2.loc[df2.index < '2014-09-24 18:00:00']['T (degC)']               # datetime before null values

# Train
train_len = round(len(train_val_temps) * 0.7)
temps_of_train = train_val_temps[10 :train_len]


# Test
temps_of_test = df2[df2.index > '2016-10-28 11:00:00']['T (degC)']                # datetime after null values                 ***# Test Target is done and ready
temps_of_test = temps_of_test[10 :]

# Validation
val_len = round(len(train_val_temps) * 0.3)
temps_of_validation = train_val_temps[-val_len + 10:]


# -------------------  Web Page
st.title("Data Science in Action \n")



 # Evaluation function in MSE 
def get_mse(y_true, preds):
    return round(mean_squared_error(y_true, preds), 3)



#    PLOTS


st.sidebar.title("Filters")

predictions_to_show = st.sidebar.selectbox('Predictions sample', ["None", "Train", "Validation", "Test"])
quarter_to_plot = st.sidebar.selectbox('Quarter', ['All', 'First', 'Second', 'Third', 'Fourth'])

if predictions_to_show == 'None':

    st.write("#### Hello world...    👋")

    st.write("Welcome to my Time Series Analysis webpage. Here I demonstrate temperature forecasting with LSTM model. An example of a Recurrent Neural Network.")

    st.write("\nBelow is how the climate dataset looks like. It will be split into training, validation and test samples. Afterwards, it will be preprocessed and analyzed. Then the LSTM model will be trained on the data to finally make predictions.")



    st.write(data.head())
    st.write(f'The shape of the data is {data.shape}')
    st.write('\n')



#  PLOT THE TEMPERATURE ON LANDING PAGE

    landing_page_T_plot = df2.resample('d').mean()

    from Plot_functions import all_samples_plot_quarter


    all_samples_plot_quarter(quarter_to_plot, landing_page_T_plot)





#------------------------------     OTHER PAGES      ----------------------------------

                # ***       TRAIN PAGE

elif predictions_to_show == 'Train':

# filter by years for the plots
    train_year = st.sidebar.selectbox('Year', ["All", "2009", "2010", "2011", "2012", "2013"])



    
    st.write("After training the model on the training set, the evaluation will be performed on all the Train, Validation, and the Test sets using the Mean Squared Error. We do keep in mind however, that the predictions will be much better on the training set because the model has already seen the data.")



    st.write("The Mean Squared Error of the train predictions is  ", get_mse(temps_of_train, train_preds['Predicted T (degC)']))




    st.write("\n \n")


    # ------------------------------------------------------



    #  Prepare the Train data for plotting
    train_plot_df = pd.DataFrame({'True Temperature' : temps_of_train, 'Predicted Temperature' : train_preds['Predicted T (degC)'] })
    train_plot_df['Date'] = pd.to_datetime(train_plot_df.index, format = 'mixed')

    if train_year != "All":
        train_plot_df_1 = train_plot_df[train_plot_df.Date.dt.year == int(train_year)]

        from Plot_functions import train_plot_quarter

        train_plot_quarter(quarter_to_plot, train_plot_df_1)



    else:
        from Plot_functions import train_plot_quarter

        train_plot_quarter(quarter_to_plot, train_plot_df)


elif predictions_to_show == 'Validation':



# VALIDATION
    from Plot_functions import validation_plot_quarter

    st.write("Now that we have evaluated the Train predictions, the validation sample will also be evaluated and it will give us a better understanding of how well the model performs in relation to the train predictions.")



    st.write("The Mean Squared Error of the Validation predictions is  ", get_mse(temps_of_validation, val_preds['Predicted T (degC)']))
   


    #  Prepare the Validation data for plotting
    val_plot_df = pd.DataFrame({'True Temperature' : temps_of_validation, 'Predicted Temperature' : val_preds['Predicted T (degC)'] })
    val_plot_df['Date'] = pd.to_datetime(val_plot_df.index, format = 'mixed')

    val_year = st.sidebar.selectbox('Year', ["All", "2013", "2014"])


    if val_year != "All":
        val_plot_df_1 = val_plot_df[val_plot_df.Date.dt.year == int(val_year)]
        

        validation_plot_quarter(quarter_to_plot, val_plot_df_1)



    else:       
        validation_plot_quarter(quarter_to_plot, val_plot_df)


else:
# TEST

    st.write("Now to see our model performance on data it has not seen, we will evaluate the Test predictions against true Test values.")
    st.write("The Mean Squared Error of the Test predictions is  ", get_mse(temps_of_test, test_preds['Predicted T (degC)']))



    #  Prepare the Test data for plotting
    test_plot_df = pd.DataFrame({'True Temperature' : temps_of_test, 'Predicted Temperature' : test_preds['Predicted T (degC)'] })


    # Create the figure and axis
    fig = go.Figure()

    # Plot the true temperature
    fig.add_trace(go.Scatter(
        x=test_plot_df.index,
        y=test_plot_df['True Temperature'],
        mode='lines',
        name='True Temperature',
        line=dict(color='blue')  
    ))

    # Plot the predicted temperature
    fig.add_trace(go.Scatter(
        x=test_plot_df.index,
        y=test_plot_df['Predicted Temperature'],
        mode='lines',
        name='Predicted Temperature',
        line=dict(color='red')  
    ))

    # Set the layout
    fig.update_layout(
        title='Model Performance on Test sample',
        xaxis_title='Datetime',
        yaxis_title='T (degC)',
        legend_title_text='',
        font=dict(
                    family="Courier New, monospace",
                    size=14,
                    color="RebeccaPurple"
                ), 
        height = 600,
        width = 1600,                         
    )

    # Apply tight layout
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width = False)


