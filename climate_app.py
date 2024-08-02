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

st.write("#### Hello world 👋")

st.write("Welcome to my Time Series Analysis page. Here I demonstrate temperature forecasting with LSTM model. An example of a Recurrent Neural Network.")

st.write("\nBelow is how the climate dataset looks like. It will be split into training, validation, test sets; preprocessed and analyzed. Then the LSTM model will be trained on the data to then make predictions.")



st.write(data.head())


 
 # Evaluation metric 
def get_mse(y_true, preds):
    return round(mean_squared_error(y_true, preds), 3)


st.write("After training the model on the Training set, the evaluation will be performed on the validation and the Test sets, using the Mean Squared Error.")


# Here I will test if the function works well
#st.write("The Average Squared Error of the train predictions is  ", get_mse(temps_of_train, train_preds['Predicted T (degC)']))
st.write("The Mean Squared Error of the Validation predictions is  ", get_mse(temps_of_validation, val_preds['Predicted T (degC)']))
st.write("The Mean Squared Error of the Test predictions is  ", get_mse(temps_of_test, test_preds['Predicted T (degC)']))



# filter by years for the plots
train_year = st.sidebar.selectbox('Train Year', ["All", "2009", "2010", "2011", "2012", "2013"])
val_year = st.sidebar.selectbox('Validation Year', ["All", "2013", "2014"])



st.write("\n, \n")


# ------------------------------------------------------
#    PLOTS


#  Prepare the Train data for plotting
train_plot_df = pd.DataFrame({'True Temperature' : temps_of_train, 'Predicted Temperature' : train_preds['Predicted T (degC)'] })
train_plot_df['Date'] = pd.to_datetime(train_plot_df.index, format = 'mixed')

if train_year != "All":
    train_plot_df_1 = train_plot_df[train_plot_df.Date.dt.year == int(train_year)]

    fig = go.Figure()

    # Plot the true temperature
    fig.add_trace(go.Scatter(
        x = train_plot_df_1.index,
        y = train_plot_df_1 ['True Temperature'],
        mode='lines',
        name='True Temperature',
        line=dict(color='blue') 
    ))

    # Plot the predicted temperature
    fig.add_trace(go.Scatter(
        x = train_plot_df_1['Date'],
        y = train_plot_df_1['Predicted Temperature'],
        mode='lines',
        name='Predicted Temperature',
        line=dict(color='orange') 
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
                ),  width = 1900, 
                    height = 600  

            )

    # Apply tight layout
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    # Display the plot in Streamlit
    st.plotly_chart(fig,  use_container_width=False)
else:
    fig = go.Figure()

    # Plot the true temperature
    fig.add_trace(go.Scatter(
        x = train_plot_df.index,
        y = train_plot_df['True Temperature'],
        mode='lines',
        name='True Temperature',
        line=dict(color='blue') 
    ))

    # Plot the predicted temperature
    fig.add_trace(go.Scatter(
        x = train_plot_df.index,
        y = train_plot_df['Predicted Temperature'],
        mode='lines',
        name='Predicted Temperature',
        line=dict(color='orange') 
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
                ),  width = 1900, 
                    height = 600  

            )

    # Apply tight layout
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    # Display the plot in Streamlit
    st.plotly_chart(fig,  use_container_width=False)


st.write("\n, \n")





# # Create the figure and axis
# fig, ax = plt.subplots(figsize=(13, 7))

# # Plot the data on the same axis
# ax.plot(train_plot_df['True Temperature'], label='True Temperature')
# ax.plot(train_plot_df['Predicted Temperature'], label='Predicted Temperature')

# # Add labels and legend
# ax.set_xlabel('Datetime')
# ax.set_ylabel('T (degC)')
# ax.set_title('Model Performance on Train sample')
# ax.legend()

# # Display the plot in Streamlit
# st.pyplot(fig)


# st.write("\n, \n")



        # -------------------


# VALIDATION

st.write("Now that we have evaluated the Train predicions, the validation sample will also be evaluated and we will give us a better understanding of how well the model performs relative to the train predictions.")


#  Prepare the Validation data for plotting
val_plot_df = pd.DataFrame({'True Temperature' : temps_of_validation, 'Predicted Temperature' : val_preds['Predicted T (degC)'] })
val_plot_df['Date'] = pd.to_datetime(val_plot_df.index, format = 'mixed')

if val_year != "All":
    val_plot_df_1 = val_plot_df[val_plot_df.Date.dt.year == int(val_year)]
    
 
    fig = go.Figure()

    # Plot the true temperature
    fig.add_trace(go.Scatter(
        x = val_plot_df_1.index,
        y = val_plot_df_1['True Temperature'],
        mode='lines',
        name='True Temperature',
        line=dict(color='blue') 
    ))

    # Plot the predicted temperature
    fig.add_trace(go.Scatter(
        x = val_plot_df_1.index,
        y = val_plot_df_1['Predicted Temperature'],
        mode='lines',
        name='Predicted Temperature',
        line=dict(color='red') 
    ))

    # Set the layout
    fig.update_layout(
        title='Model Performance on Validation sample',
        xaxis_title='Datetime',
        yaxis_title='T (degC)',
        legend_title_text='',
        font=dict(
            family="Courier New, monospace",
            size=14,
            color="RebeccaPurple"
                ),  width = 1900, 
                    height = 600  

            )

    # Apply tight layout
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    # Display the plot in Streamlit
    st.plotly_chart(fig,  use_container_width=False)
else:
    fig = go.Figure()

    # Plot the true temperature
    fig.add_trace(go.Scatter(
        x = val_plot_df['Date'],
        y = val_plot_df ['True Temperature'],
        mode='lines',
        name='True Temperature',
        line=dict(color='blue') 
    ))

    # Plot the predicted temperature
    fig.add_trace(go.Scatter(
        x = val_plot_df.index,
        y = val_plot_df['Predicted Temperature'],
        mode='lines',
        name='Predicted Temperature',
        line=dict(color='orange') 
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
                ),  width = 1900, 
                    height = 600  

            )

    # Apply tight layout
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

    # Display the plot in Streamlit
    st.plotly_chart(fig,  use_container_width=False)



st.write("\n, \n")

#--------------------


# TEST

st.write("Now to see our model performance on data it has not seen, we will evaluate the Test predictions against true Test values.")


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
    width = 1600,                          # Run it !!!
)

# Apply tight layout
fig.update_layout(margin=dict(l=20, r=20, t=30, b=20))

# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width = False)


