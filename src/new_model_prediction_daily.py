import os
import modal
from PIL import Image
import requests
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.pyplot import figure
import numpy as np


LOCAL=True

if LOCAL == False:
   stub = modal.Stub("prediction_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "lxml", "joblib", "urllib3", "jsonschema", "xgboost", "scikit-learn"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()
       
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
    elif snow_level > 40 and snow_level <= 60:
        emoji = emojis[1]
    elif snow_level > 60:
        emoji = emojis[0]

    return emoji
    
def plot_snow_prediction(pred_snow):
    '''
    Build fancy histogram for snow levels.
    Format dates and plot bar chart.
    '''
    # format dates
    old_format = '%Y-%m-%d'
    #new_format = '%a %d, %b'
    new_format = '%d %b'
    for elem in pred_snow['time']:
        #print("Date: ", elem)
        new_elem = datetime.datetime.strptime(elem, old_format)
        new_elem = new_elem.strftime(new_format)
        #print("Format: ", new_elem)
        pred_snow['time'].replace(to_replace=elem, value=new_elem, inplace=True)

    # plot bar chart
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
    
def plot_accuracy_graph(pred_now, actual_snow):
    '''
    Get the actual snow of past two weeks and predicted snow
    of next six days + past 8 predicted to make comparison.
    Save plot to be uploaded in Hopsworks, used in app in
    a second tab.
    Plot also vertical line to indicate today.
    '''
    
    return
       
def build_pictures_for_app(project, pred_snow):
    '''
    Create a plot and six emojis to store in Hopsworks
    for later use in the Huggingface app
    '''
    dataset_api = project.get_dataset_api()
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
    
    plot_snow_prediction(pred_snow)
    dataset_api.upload("./images/img_pred/plot.png", "Resources/img_prediction", overwrite=True)
    print("Uploaded pictures.")
    
    return
    
    
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
    print("Forecast:\n", forecast_df)

    forecast_df = forecast_df.sort_values(by=["time"], ascending=[True]).reset_index(drop=True)
    forecast_df = forecast_df.drop(columns=["time"]).fillna(0)
    pred = model.predict(forecast_df)
    print(pred)

    pred_df['snow_level_prediction'] = pred
    print(pred_df)


    snow_predictions_fg = fs.get_or_create_feature_group(
    name="snow_predictions",
    version=1,
    primary_key=["time"], 
    description="Snow level predictions")
    snow_predictions_fg.insert(pred_df, write_options={"wait_for_job" : False})
    
    # create pictures for the app in Huggingface
    build_pictures_for_app(project, pred_df)
    #plot_snow_prediction(pred_df)
    
    return


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("prediction_daily")
        with stub.run():
            f()
