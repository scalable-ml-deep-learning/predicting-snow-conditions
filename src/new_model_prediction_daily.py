import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("prediction_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "lxml", "joblib", "urllib3", "jsonschema", "xgboost", "scikit-learn"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()
       
def build_pictures_for_app(project, pred_snow):
    '''
    Create a plot and six emojis to store in Hopsworks
    for later use in the Huggingface app
    '''
    emojis = ["cool-face", "heart-eyes", "light-smile", "unamused"]
    
    # create ranges to choose emoji
    for index in range(len(pred_snow)):
        snow = pred_snow['snow_level_prediction']
        if snow < 60:
            # enough snow
            picture = emojis[0]
            img_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/predicting-snow-conditions/feature-tommaso/src/images/" + picture + ".png"
            img = Image.open(requests.get(passenger_url, stream=True).raw)
            dataset_api = project.get_dataset_api()
            dataset_api.upload(img, "Resources/img_prediction", overwrite=True)
    '''
    for picture in pictures:
        #passenger_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/predicting-snow-conditions/feature-tommaso/src/images/" + passenger + ".png"
        #img = Image.open(requests.get(passenger_url, stream=True).raw)  
        #passenger_path =  "../img/" + str(passenger) + ".png"
        #img = Image.open(passenger_path, mode='r')
           
        #img.save("./images/"+ picture + ".png")
        dataset_api = project.get_dataset_api()
        dataset_api.upload("./images/"+ picture + ".png", "Resources/img_emojis", overwrite=True)
    '''   
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
    model = mr.get_model("snow_model", version=1)
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
    
    return


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("prediction_daily")
        with stub.run():
            f()