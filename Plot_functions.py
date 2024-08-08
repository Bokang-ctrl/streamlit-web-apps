import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


#  IF THE SAMPLES CHOSEN IS FOR ALL SAMPLES
def all_samples_plot_quarter(quarter, dataframe):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = dataframe.index,
            y = dataframe['T (degC)'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))

        # Set the layout
        fig.update_layout(
            title='Average Temperature (2009 - 2017)',
            xaxis_title='Date',
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
    
    
    
    
    elif quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['T (degC)'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))

        # Set the layout
        fig.update_layout(
            title='Average Temperature (1st Quarter of the sample)',
            xaxis_title='Date',
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
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['T (degC)'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))

        # Set the layout
        fig.update_layout(
            title='Average Temperature (2nd Quarter of the sample)',
            xaxis_title='Date',
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


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['T (degC)'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))

        # Set the layout
        fig.update_layout(
            title='Average Temperature (3rd Quarter of the sample)',
            xaxis_title='Date',
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
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['T (degC)'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))

        # Set the layout
        fig.update_layout(
            title='Average Temperature (4th Quarter of the sample)',
            xaxis_title='Date',
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
        
    


        #  IF THE SAMPLE CHOSEN IS THE TRAIN AND HAS YEAR
def train_plot_quarter_with_year(quarter, dataframe, train_year):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = dataframe.index,
            y = dataframe['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title= f'Train predictions vs True Temperatures ({train_year})',
            xaxis_title='Date',
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
    
    
    
    
    
    
    elif quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title= f'Train predictions vs True Temperatures (1st Quarter of {train_year})',
            xaxis_title='Date',
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
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        fig = go.Figure()

        # Plot the data temperatureo
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title= f'Train predictions vs True Temperatures (2nd Quarter of {train_year})',
            xaxis_title='Date',
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


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title = f'Train predictions vs True Temperatures (3rd Quarter of {train_year})',
            xaxis_title='Date',
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
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title= f'Train predictions vs True Temperatures (4th Quarter of {train_year})',
            xaxis_title='Date',
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
        


#  IF THE SAMPLE CHOSEN IS THE TRAIN AND HAS NO YEAR


def train_plot_quarter_no_year(quarter, dataframe):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = dataframe.index,
            y = dataframe['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures',
            xaxis_title='Date',
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
    
    
    
    
    
    
    elif quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures (1st Quarter of the sample)',
            xaxis_title='Date',
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
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        fig = go.Figure()

        # Plot the data temperatureo
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures (2nd Quarter of the sample)',
            xaxis_title='Date',
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


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures (3rd Quarter of the sample)',
            xaxis_title='Date',
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
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures (4th Quarter of the sample)',
            xaxis_title='Date',
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
        






# VALIDATION WITH YEAR

def validation_plot_quarter_with_year(quarter, dataframe, val_year):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = dataframe.index,
            y = dataframe['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures ({val_year})',
            xaxis_title='Date',
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
    
    
    
    
    
    
    elif quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures (1st Quarter of {val_year})',
            xaxis_title='Date',
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
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures (2nd Quarter of {val_year})',
            xaxis_title='Date',
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


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures (3rd Quarter of {val_year})',
            xaxis_title='Date',
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
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures (4th Quarter of the {val_year})',
            xaxis_title='Date',
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
        


#       VALIDATION WITH NO YEAR
def validation_plot_quarter_no_year(quarter, dataframe):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = dataframe.index,
            y = dataframe['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures',
            xaxis_title='Date',
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
    
    
    
    
    
    
    elif quarter == 'First':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[:q_length]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title= f'Validation predictions vs True Temperatures (1st Quarter of the sample)',
            xaxis_title='Date',
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
        
        
    
    elif quarter == 'Second':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length : q_length * 2]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures (2nd Quarter of the sample)',
            xaxis_title='Date',
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


    elif quarter == 'Third':
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 2 : q_length * 3]
        
    
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures (3rd Quarter of the sample)',
            xaxis_title='Date',
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
        q_length = round(len(dataframe) * q_len)
        actual_quarter_df = dataframe.iloc[q_length * 3 : ]
        
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['True Temperature'],
            mode='lines',
            name='Temperature (degC)',
            line=dict(color='blue') 
        ))
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='orange') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures (4th Quarter of the sample)',
            xaxis_title='Date',
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
        
    