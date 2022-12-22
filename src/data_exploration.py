import numpy as np
import requests
import matplotlib.pyplot as plt
import hopsworks
import joblib
import pandas as pd

def sort_by_time(df):
    '''
    Sort dataframe by time
    '''
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', ascending=True, inplace=True)
    return df

def get_weather():
    '''
    Get weather data from feature group
    '''
    
    weather_fg = fs.get_feature_group(name="weather_data")
    weather = weather_fg.read()
    # sort the dataframe by date
    weather = sort_by_time(weather)
    
    return weather
    
def plot_weather(weather):
    '''
    Plot weather forecast and history.
    '''
    time = weather['time'] # used for x-axis
    for feature in weather.keys():
        if feature == 'time' or feature == 'weathercode':
            pass
        else:
          print("Feature: ", feature)
          #print(weather[feature])
          plt.plot(time, weather[feature], label=f'{feature}')
        
    return

def get_actual_snow():
    '''
    Get from Hopsworks the actual snow level day by day
    since the beginning of the collection.
    '''
    
    snow_data_fg = fs.get_feature_group(name="snow_data")
    snow_data = snow_data_fg.read()
    # sort by time
    snow_data = sort_by_time(snow_data)
    
    return snow_data
    
def plot_actual_snow(actual_snow):
    '''
    Plot the actual snow level vs time.
    '''
    plt.plot(actual_snow['time'], actual_snow['hs'], label='actual snow', linestyle=":")
    return
    
def get_snow_prediction():
    '''
    Get the latest prediction from Hopsworks.
    Returns the prediction.
    '''

    prediction_fg = fs.get_feature_group(name="snow_predictions", version=1)
    prediction = prediction_fg.read()
    prediction = sort_by_time(prediction)
    
    return prediction
    
def plot_predicted_snow(predicted_snow):
    '''
    Plot the predicted snow level vs time.
    '''
    plt.plot(predicted_snow['time'], predicted_snow['snow_level_prediction'], label='predicted snow', linestyle="--")
    return
    
if __name__ == '__main__':
    project = hopsworks.login(project="finetune")
    fs = project.get_feature_store()
    
    weather = get_weather()
    actual_snow = get_actual_snow()
    predicted_snow = get_snow_prediction()
    plot_weather(weather)
    plot_actual_snow(actual_snow)
    plot_predicted_snow(predicted_snow)
    plt.legend()
    plt.show()
    

