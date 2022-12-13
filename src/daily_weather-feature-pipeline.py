import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("weather_data_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "urllib3", "jsonschema"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    from urllib.request import urlopen
    import json
    import pandas as pd

    project = hopsworks.login(project="finetune")
    fs = project.get_feature_store()

    url = "https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto"  
    # store the response of URL
    response = urlopen(url)
    # storing the JSON response from url in data
    data_json = json.loads(response.read())

    #################
    # DATA CLEANING #
    #################
    # convert dictionary to dataframe and select daily key
    weather_df = pd.DataFrame.from_dict(data_json['daily'], orient='columns')
    print("Historical weather:\n", weather_df)

    weather_df = weather_df.iloc[:1]
    print(weather_df)
    
    snow_fg = fs.get_feature_group(name="weather_data",version=1)
    snow_fg.insert(weather_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("weather_data_daily")
        with stub.run():
            f()
