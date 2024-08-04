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
            title='Average Temperature (2009 - 2016)',
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
            title='Average Temperature (2009 - 2016)',
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
            title='Average Temperature (2009 - 2016)',
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
            title='Average Temperature (2009 - 2016)',
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
            title='Average Temperature (2009 - 2016)',
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
        
    


        #  IF THE SAMPLE CHOSEN IS THE TRAIN
def train_plot_quarter(quarter, dataframe):
    q_len = 0.25
    
    
    if quarter == 'All':
        fig = go.Figure()

        # Plot the data temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df.index,
            y = actual_quarter_df['T (degC)'],
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
            line=dict(color='red') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures Q1',
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
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures Q2',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures Q3',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Train predictions vs True Temperatures Q4',
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
        



def validation_plot_quarter(quarter, dataframe):
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = dataframe['Date'],
            y = dataframe['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        ))
        
        

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        ))      
        
        
        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures Q1',
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
        
    
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        ))   
    

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures Q2',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures Q3',
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
        
      # Plot the predicted temperature
        fig.add_trace(go.Scatter(
            x = actual_quarter_df['Date'],
            y = actual_quarter_df['Predicted Temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red') 
        )) 
        
        

        # Set the layout
        fig.update_layout(
            title='Validation predictions vs True Temperatures Q4',
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
        
    