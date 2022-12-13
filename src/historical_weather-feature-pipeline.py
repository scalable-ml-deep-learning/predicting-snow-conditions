import os
import pandas as pd
import hopsworks
from urllib.request import urlopen
import json
import time
import datetime

#project = hopsworks.login(project="finetune")
#fs = project.get_feature_store()

url = "https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-12"  
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
print("Time:\n", weather_df['time'])
# convert dates to timestamp to prepare join
for value in  weather_df['time'].values:
  print("Value: ", value)
  new_value = time.mktime(datetime.datetime.strptime(value, "%Y-%m-%d").timetuple())
  print("New value: ", new_value)
  weather_df.replace(to_replace=value, value=new_value, inplace = True)

print("New historical weather:\n", weather_df)
'''
weather_fg = fs.get_or_create_feature_group(
    name="weather_data",
    version=1,
    primary_key=["time"], 
    description="Weather dataset")
weather_fg.insert(weather_df, write_options={"wait_for_job" : False})
'''
