import os
import hopsworks
import pandas as pd

#project = hopsworks.login(project="finetune")
#fs = project.get_feature_store()

weather_df = pd.read_json("https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&hourly=temperature_2m,snow_depth,weathercode,visibility&models=best_match&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-10")

weather_df.drop(['latitude', 'longitude','hourly_units', 'generationtime_ms', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation', 'elevation', 'current_weather'], axis='columns', inplace=True)
weather_df.dropna(inplace=True)
#print(weather_df.keys)

weather_df = weather_df['hourly']
for key in weather_df.keys():
    df_ = []
    for i in range(0, len(weather_df[key]), 24):
        df_.append(weather_df[key][i])
    weather_df[key] = df_

#weathercode_df = weather_df['weathercode']
#time_df = weather_df['time']


print(weather_df)

#snow_fg = fs.get_or_create_feature_group(
#    name="snow_conditions",
#    version=1,
#    primary_key=["dataMis"], 
#    description="Snow conditions dataset")
#snow_fg.insert(snow_df, write_options={"wait_for_job" : False})