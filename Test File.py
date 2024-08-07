### PERFECT!!!!    tHE MAIN CLIMATE APP FILE SHOULD BE SIMILAR BUT WITH GO PLOTS FROM PLOTLY....



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error






#@st.cache_data
def load_datasets():
    # Load the datasets
    data = pd.read_csv('jena_climate_2009_2016.csv')
    train_preds = pd.read_csv('train_predictions.csv')
    test_preds = pd.read_csv('test_predictions.csv')
    val_preds = pd.read_csv('validation_predictions.csv')

    return data, train_preds, test_preds, val_preds

#Load the datasets using the cached function
data, train_preds, test_preds, val_preds = load_datasets()


# Data Preprocessing

df = data.copy()
df['Date Time'] = pd.to_datetime(df['Date Time'], format = "mixed")
df.set_index('Date Time', inplace = True)
df1 = df.resample('h').mean()
df2 = df1.drop(columns = ['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])


# set the date column as index
train_preds.set_index('Date', inplace = True)
train_preds.index = pd.to_datetime(train_preds.index)

test_preds.set_index('Date', inplace = True)
test_preds.index = pd.to_datetime(test_preds.index)

val_preds.set_index('Date', inplace = True)
val_preds.index = pd.to_datetime(val_preds.index)




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



 # Evaluation function in MSE 
def get_mse(y_true, preds):
    return round(mean_squared_error(y_true, preds), 3)


#%%
#    PLOTS

####### start run here

predictions_to_show = input(f'Choose prediction sample: \n   None, Train, Validation, Test: ')
quarter_to_plot = input(f" Choose Quarter sample between: \n    All, First, Second, Third, Fourth \n --- ")

q_len = 0.25

if predictions_to_show == 'None':


#  PLOT THE TEMPERATURE ON LANDING PAGE

    landing_page_T_plot = df2.resample('d').mean()


    if quarter_to_plot == 'All':
    
        plt.figure(figsize = (12, 8))
        sns.lineplot(data = landing_page_T_plot, x = landing_page_T_plot.index, y = 'T (degC)', color = 'blue')
        plt.title('Average Temperature (2009 - 2017)')
        plt.show()
        

    
    
    
    
    elif quarter_to_plot == 'First':
        q_length = round(len(landing_page_T_plot) * q_len)
        actual_quarter_df = landing_page_T_plot.iloc[:q_length]
        
        plt.figure(figsize = (12, 8))
        sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y ='T (degC)', color = 'blue')
        plt.title('Quarter 1 of sample')
        plt.show()

    
    elif quarter_to_plot == 'Second':
        q_length = round(len(landing_page_T_plot) * q_len)
        actual_quarter_df = landing_page_T_plot.iloc[q_length : q_length * 2]
        
        plt.figure(figsize = (12, 8))
        sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'T (degC)', color = 'blue')
        plt.title('Quarter 2 of sample')
        plt.show()
        

    elif quarter_to_plot == 'Third':
        q_length = round(len(landing_page_T_plot) * q_len)
        actual_quarter_df = landing_page_T_plot.iloc[q_length * 2 : q_length * 3]
        
        plt.figure(figsize = (12, 8))
        sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'T (degC)', color = 'blue')
        plt.title('Quarter 3 of sample')
        plt.show()

        

    else:
        q_length = round(len(landing_page_T_plot) * q_len)
        actual_quarter_df = landing_page_T_plot.iloc[q_length * 3 : ]
        
        plt.figure(figsize = (12, 8))
        sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'T (degC)', color = 'blue')
        plt.title('Quarter 4 of sample')
        plt.show()
        
    



#------------------------------     OTHER PAGES      ----------------------------------

                # ***       TRAIN PAGE

elif predictions_to_show == 'Train':

# filter by years for the plots
    train_year = input("Choose Year between \n All, 2009, 2010, 2011, 2012, 2013 : \n ---")
    



    #  Prepare the Train data for plotting
    train_plot_df = pd.DataFrame(data = {'True Temperature' : temps_of_train.values, 'Predicted Temperature' : train_preds['Predicted T (degC)']}, index = temps_of_train.index)
    train_plot_df['Date'] = pd.to_datetime(train_plot_df.index, format = 'mixed')

   
    


    if train_year != "All":
        train_plot_df_1 = train_plot_df[train_plot_df.Date.dt.year == int(train_year)]
        
        len_quarter_df =   round(len(train_plot_df_1) * q_len)
        
        ###   Quarter if statements
        if quarter_to_plot == 'All':        
 
            
 
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = train_plot_df_1, x = train_plot_df_1.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = train_plot_df_1, x = train_plot_df_1.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year}) ')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'First':    
            actual_quarter_df = train_plot_df_1[: len_quarter_df]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'Second':    
            actual_quarter_df = train_plot_df_1[len_quarter_df : len_quarter_df * 2]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
        elif quarter_to_plot == 'Third':    
            actual_quarter_df = train_plot_df_1[len_quarter_df  * 2: len_quarter_df * 3]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            


        else:
            actual_quarter_df = train_plot_df_1[len_quarter_df  * 3: ]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = "Predictions")
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
            

    else:               # This is if train_year == ALL
        len_quarter_df =   round(len(train_plot_df) * q_len)
        
        ###   Quarter if statements
        if quarter_to_plot == 'All':        
 
            
 
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = train_plot_df, x = train_plot_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = train_plot_df, x = train_plot_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions) ')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'First':    
            actual_quarter_df = train_plot_df[: len_quarter_df]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'Second':    
            actual_quarter_df = train_plot_df[len_quarter_df : len_quarter_df * 2]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
        elif quarter_to_plot == 'Third':    
            actual_quarter_df = train_plot_df[len_quarter_df  * 2: len_quarter_df * 3]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            


        else:
            actual_quarter_df = train_plot_df[len_quarter_df  * 3: ]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = "Predictions")
            plt.title(f'Train Sample Temperatures vs Predictions  ({train_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            

                


elif predictions_to_show == 'Validation':



# VALIDATION


    #  Prepare the Validation data for plotting
    val_plot_df = pd.DataFrame({'True Temperature' : temps_of_validation.values, 'Predicted Temperature' : val_preds['Predicted T (degC)']}, index = temps_of_validation.index)
    val_plot_df['Date'] = pd.to_datetime(val_plot_df.index, format = 'mixed')

    val_year = input('Choose Year : \n     All, 2013, 2014 ---')


    len_quarter_df =   round(len(val_plot_df_1) * q_len)
    
    if val_year != "All":
        val_plot_df_1 = val_plot_df[val_plot_df.Date.dt.year == int(val_year)]
        
        
        
        ###   Quarter if statements
        if quarter_to_plot == 'All':        
 
            
 
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = val_plot_df_1, x = val_plot_df_1.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = val_plot_df_1, x = val_plot_df_1.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions  ({val_year}) ')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'First':    
            actual_quarter_df = val_plot_df_1[: len_quarter_df]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions  ({val_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'Second':    
            actual_quarter_df = val_plot_df_1[len_quarter_df : len_quarter_df * 2]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions  ({val_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
        elif quarter_to_plot == 'Third':    
            actual_quarter_df = val_plot_df_1[len_quarter_df  * 2: len_quarter_df * 3]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions  ({val_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            


        else:
            actual_quarter_df = val_plot_df_1[len_quarter_df  * 3: ]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = "Predictions")
            plt.title(f'Validation Sample Temperatures vs Predictions  ({val_year})   {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
                




    else:
        

        if quarter_to_plot == 'All':        
 
            
 
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = val_plot_df, x = val_plot_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = val_plot_df, x = val_plot_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions) ')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'First':    
            actual_quarter_df = val_plot_df[: len_quarter_df]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions    {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
            
        elif quarter_to_plot == 'Second':    
            actual_quarter_df = val_plot_df[len_quarter_df : len_quarter_df * 2]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions    {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            
        elif quarter_to_plot == 'Third':    
            actual_quarter_df = val_plot_df[len_quarter_df  * 2: len_quarter_df * 3]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = 'Prediction')
            plt.title(f'Validation Sample Temperatures vs Predictions     {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            
            


        else:
            actual_quarter_df = val_plot_df[len_quarter_df  * 3: ]
            
            plt.figure(figsize = (12, 8))
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'True Temperature', color = 'blue', label = 'True Temperature (degC')
            sns.lineplot(data = actual_quarter_df, x = actual_quarter_df.index, y = 'Predicted Temperature', color = 'red', label = "Predictions")
            plt.title(f'Validation Sample Temperatures vs Predictions     {quarter_to_plot} quarter of sample')
            plt.legend()
            plt.show()
            











else:
# TEST


    #  Prepare the Test data for plotting
    test_plot_df = pd.DataFrame(data = {'True Temperature' : temps_of_test.values, 'Predicted Temperature' : test_preds['Predicted T (degC)']}, index = temps_of_test.index)

    test_plot_df['Predicted Temperature'] = test_preds['Predicted T (degC)']



    plt.figure(figsize = (12, 8))
    sns.lineplot(data = test_plot_df, x = test_plot_df.index, y = 'True Temperature', color = 'blue')
    sns.lineplot(data = test_plot_df, x = test_plot_df.index, y = 'Predicted Temperature', color = 'red')
    plt.title('Test sample predictions')
    plt.show()
