import os
import pandas as pd
import hopsworks
from urllib.request import urlopen
import json

#project = hopsworks.login(project="finetune")
#fs = project.get_feature_store()

url = "https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-12"  
# store the response of URL
response = urlopen(url)
# storing the JSON response from url in data
data_json = json.loads(response.read())
# convert dictionary to dataframe
weather_df = pd.DataFrame.from_dict(data_json['daily'], orient='columns')

print("Historical weather:\n", weather_df)

'''
weather_fg = fs.get_or_create_feature_group(
    name="weather_data",
    version=1,
    primary_key=["time"], 
    description="Weather dataset")
weather_fg.insert(weather_df, write_options={"wait_for_job" : False})
'''
