import os
import modal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import os
from PIL import Image
from datetime import date, datetime, timedelta


LOCAL=False

if LOCAL == False:
   stub = modal.Stub("prediction_daily_v2")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "lxml", "joblib", "urllib3", "jsonschema", "xgboost", "scikit-learn", "matplotlib"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()

################
# SELECT EMOJI #
################
def emoji_selection(snow_level):
    '''
    Select the right emoji for the snow level.
    '''
    # create ranges to choose emoji
    emojis = ["heart-eyes", "cool-face", "light-smile", "unamused"]
    if snow_level <= 20:
        emoji = emojis[3]
    elif snow_level > 20 and snow_level <= 40:
        emoji = emojis[2]
    elif snow_level > 40 and snow_level <= 65:
        emoji = emojis[1]
    elif snow_level > 65:
        emoji = emojis[0]

    return emoji
    
###############
# FORMAT TIME #
###############
def format_time(df_time):
    '''
    Format dates given dataframe.
    '''
    # format dates
    old_format = '%Y-%m-%d'
    #new_format = '%a %d, %b'
    new_format = '%d %b'
    for elem in df_time['time']:
        #print("Date: ", elem)
        new_elem = datetime.strptime(elem, old_format)
        new_elem = new_elem.strftime(new_format)
        #print("Format: ", new_elem)
        df_time['time'].replace(to_replace=elem, value=new_elem, inplace=True)
    
    return df_time

##########################
# SORT DATAFRAME BY TIME #
##########################
def sort_by_time(df):
    '''
    Sort dataframe by time
    '''
    #df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', ascending=True, inplace=True)
    return df

########################
# PLOT SNOW PREDICTION #
########################
def plot_snow_prediction(pred_snow):
    '''
    Build fancy histogram for snow levels of next six days.
    Format dates and plot bar chart.
    '''
    # format dates
    pred_snow = format_time(pred_snow)
    # plot bar chart
    plt.figure()
    plt.bar(
        pred_snow['time'], 
        pred_snow['snow_level_prediction'], 
        color = 'hotpink',
        )
    plt.ylabel("Centimeters of snow") 
    plt.title("Snow level forecast for Passo Rolle (TN), Italy")
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    #plt.show()
    plt.savefig('./images/img_pred/plot.png')
    
    return

#############################
# GET ACTUAL SNOW DATAFRAME #
#############################   
def get_actual_snow(fs):
    '''
    Get from Hopsworks the actual snow level day by day
    since the beginning of the collection.
    '''
    snow_data_fg = fs.get_feature_group(name="snow_data")
    snow_data = snow_data_fg.read()
    # sort by time
    snow_data = sort_by_time(snow_data)
    
    return snow_data

############################
# PLOT ACTUAL SNOW HISTORY #
############################  
def plot_actual_snow(actual_snow):
    '''
    Plot the actual snow level vs time.
    '''
    actual_snow = format_time(actual_snow)
    plt.plot(actual_snow['time'], actual_snow['hs'], label='actual snow')
    return

#################################
# GET SNOW PREDICTION DATAFRAME #
#################################    
def get_snow_prediction(fs):
    '''
    Get the latest prediction from Hopsworks.
    Returns the prediction.
    '''
    prediction_fg = fs.get_feature_group(name="snow_predictions", version=1)
    prediction = prediction_fg.read()
    # sort by time
    prediction = sort_by_time(prediction)
    
    return prediction
    
###############################
# PLOT PREDICTED SNOW HISTORY #
###############################
def plot_predicted_snow(predicted_snow):
    '''
    Plot the predicted snow level vs time.
    '''
    predicted_snow = format_time(predicted_snow)
    plt.plot(predicted_snow['time'], predicted_snow['snow_level_prediction'], label='predicted snow', linestyle=":")
    return

###################################
# RETURN LAST N DAYS OF DATAFRAME #
###################################  
def last_n_days(df, days):
    '''
    Return last days of history.
    '''
    # delete dates later than today (in case of forecast)
    yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')
    df = df[df['time'] <= yesterday]
    # select last n elements
    df = df.tail(days)
    return df

#############################################
# PLOT HISTORY OF PREDICTION VS ACTUAL SNOW #
#############################################
def plot_pred_history(project):
    '''
    Get the actual snow of past two weeks and predicted snow
    of next six days + past 8 predicted to make comparison.
    Save plot to be uploaded in Hopsworks, used in app in
    a second tab.
    Plot also vertical line to indicate today.
    '''
    # get actual and predicted snow histories
    fs = project.get_feature_store()
    actual_snow = get_actual_snow(fs)
    predicted_snow = get_snow_prediction(fs)
    # take last 10 days
    actual_snow = last_n_days(actual_snow, 10)
    predicted_snow = last_n_days(predicted_snow, 10)
    # plot both functions
    plt.figure()
    plot_actual_snow(actual_snow)
    plot_predicted_snow(predicted_snow)
    plt.ylabel("Centimeters of snow") 
    plt.title("Snow level forecast accuracy over the past 10 days")
    plt.yticks(np.arange(0, 101, 10))
    plt.xticks(rotation = 45) # Rotates X-Axis Ticks by 45-degrees
    plt.legend()
    #plt.show()
    plt.savefig('./images/img_pred/plot_history.png')
    
    return

#################################################
# PREPARE ALL PLOTS AND FIGURES FOR APPLICATION #
#################################################   
def build_pictures_for_app(project, pred_snow):
    '''
    Create a plot and six emojis to store in Hopsworks
    for later use in the Huggingface app
    '''
    dataset_api = project.get_dataset_api()
    # create emoji for each prediction day
    
    for index in range(1,len(pred_snow)+1):
        snow = pred_snow['snow_level_prediction'][index]
        # print("Snow level: ", snow)
        # select emoji
        emoji = emoji_selection(snow)
        img_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/predicting-snow-conditions/feature-tommaso/src/images/" + emoji + ".png"
        img = Image.open(requests.get(img_url, stream=True).raw)
        img.save("./images/img_pred/"+str(index)+".png")
        # upload emoji to correspondent index
        dataset_api.upload("./images/img_pred/"+str(index)+".png", "Resources/img_prediction", overwrite=True)
    # plot bar chart for prediction
    plot_snow_prediction(pred_snow)
    dataset_api.upload("./images/img_pred/plot.png", "Resources/img_prediction", overwrite=True)
    # plot history of predictions accuracy graph
    plot_pred_history(project)
    dataset_api.upload("./images/img_pred/plot_history.png", "Resources/img_prediction", overwrite=True)
    
    print("Uploaded pictures.")
    
    return
    
#################################
# MAIN FUNCTION TO RUN ON MODAL #
#################################
def g():
    import hopsworks
    import pandas as pd
    import joblib
    from urllib.request import urlopen
    import json


    EVALUATION_METRIC="mean squared error"  
    SORT_METRICS_BY="min" # your sorting criteria

    project = hopsworks.login(project="finetune")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    #model = mr.get_model("snow_model", version=1)
    # get best model based on custom metrics
    model = mr.get_best_model("snow_model",
                               EVALUATION_METRIC,
                               SORT_METRICS_BY)
    model_dir = model.download()
    model = joblib.load(model_dir + "/snow_model.pkl")
    print("Model:", model_dir)
    
    url = "https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto"  
    # store the response of URL
    response = urlopen(url)
    # storing the JSON response from url in data
    data_json = json.loads(response.read())
    # convert dictionary to dataframe and select daily key
    forecast_df = pd.DataFrame.from_dict(data_json['daily'], orient='columns')
    forecast_df = forecast_df.drop(index=0)
    pred_df = forecast_df[['time']]
    print("Weather forecast:\n", forecast_df)

    forecast_df = forecast_df.sort_values(by=["time"], ascending=[True]).reset_index(drop=True)
    forecast_df = forecast_df.drop(columns=["time"]).fillna(0)
    pred = model.predict(forecast_df)
    pred_df['snow_level_prediction'] = pred
    print("Snow prediction:\n", pred_df)


    snow_predictions_fg = fs.get_or_create_feature_group(
    name="snow_predictions",
    version=1,
    primary_key=["time"], 
    description="Snow level predictions")
    snow_predictions_fg.insert(pred_df, write_options={"wait_for_job" : False})
      
    actual_snow_fg = fs.get_feature_group(name='snow_data')
    actual_snow_df = actual_snow_fg.read()
    # create pictures for the app in Huggingface
    if not os.path.exists('./images/img_pred/'):
      os.makedirs('./images/img_pred/')
    build_pictures_for_app(project, pred_df)
    
    return


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("prediction_daily_v2")
        with stub.run():
            f()
