import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error



from sklearn.preprocessing import MinMaxScaler

# Load data
dataf = 'jena_climate_2009_2016.csv'
train = 'train_predictions.csv'
test = 'test_predictions.csv'
validation = 'validation_predictions.csv'

data = pd.read_csv(dataf)
train_preds = pd.read_csv(train)
test_preds = pd.read_csv(test)
val_preds = pd.read_csv(validation)

# Prepare DataFrame
df = data.copy()
df['Date Time'] = pd.to_datetime(df['Date Time'], format = "mixed")
df.set_index('Date Time', inplace=True)
df1 = df.resample('h').mean()


# Create non-null DataFrame 
df_no_null_1 = df1[df1.index < '2014-09-24 18:00:00']               # datetime before null values   
test_df = df1[df1.index > '2016-10-28 11:00:00']                    # datetime after null values


# Split data into input variables (X) and target variable (y)
X = df_no_null_1.drop(columns='T (degC)')
y = df_no_null_1['T (degC)']
df2 = df1.drop(columns=['p (mbar)', 'wv (m/s)', 'max. wv (m/s)', 'wd (deg)'])


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Data splitting
train_len = round(len(X_scaled) * 0.7)
val_len = round(len(X_scaled) * 0.3)

X_train, y_train = X_scaled[10 : train_len], y[10 : train_len]
X_val, y_val = X_scaled[train_len + 10:], y[train_len + 10:]

# Test data
X_test = test_df.drop(columns='T (degC)')
y_test = test_df['T (degC)'][10: ]



#%%

train_preds['Date'] = pd.to_datetime(train_preds['Date'], format = 'mixed')
val_preds['Date'] = pd.to_datetime(val_preds['Date'], format = 'mixed')


train_uniq_years = train_preds['Date'].dt.year.unique()
val_uniq_years = val_preds['Date'].dt.year.unique()


# Quarter plots function

def plot_quarter(quarter, dataframe):
    q_len = 0.25
    
    if quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        plt.figure(figsize = (12, 7))
    
        # Plot the data temperature
        sns.lineplot(data = actual_quarter_df, x = 'Date', y = 'Predicted T (degC)')
        plt.title('Temperature Plot')
        plt.show()
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        plt.figure(figsize = (12, 7))
    
        # Plot the data temperature
        sns.lineplot(data = actual_quarter_df, x = 'Date', y = 'Predicted T (degC)')
        plt.title('Temperature Plot')
        plt.show()


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
    # Plot the data temperature
        plt.figure(figsize = (12, 7))
    
        # Plot the data temperature
        sns.lineplot(data = actual_quarter_df, x = 'Date', y = 'Predicted T (degC)')
        plt.title('Temperature Plot')
        plt.show()

    else:
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
     # Plot the data temperature
        plt.figure(figsize = (12, 7))
     
         # Plot the data temperature
        sns.lineplot(data = actual_quarter_df, x = 'Date', y = 'Predicted T (degC)')
        plt.title('Temperature Plot')
        plt.show()
        
    

    
plot_quarter('Fourth', val_preds)
    
    
    
    
    

