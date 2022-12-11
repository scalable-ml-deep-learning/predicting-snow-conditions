import os
import pandas as pd
import matplotlib.pyplot as plt

weather_df = pd.read_json("https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&hourly=temperature_2m,snow_depth,weathercode,visibility&models=best_match&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-11")

weather_df.drop(['latitude', 'longitude','hourly_units', 'generationtime_ms', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation', 'elevation', 'current_weather'], axis='columns', inplace=True)
weather_df.dropna(inplace=True)

weather_df = weather_df['hourly']
for key in weather_df.keys():
    df_ = []
    for i in range(0, len(weather_df[key]), 24):
        df_.append(weather_df[key][i])
    weather_df[key] = df_

print(weather_df)

snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")

snow_df = snow_df.loc[snow_df['codStaz'] == "31RO"] # Select Passo Rolle
snow_df.drop(['codStaz', 'oraDB', 'vq1', 'vq2', 'n', 'tmin', 'tmax', 'hn', 'fi', 't10', 't30', 'pr', 'cs', 's', 'b'], axis='columns', inplace=True)
snow_df.dropna(inplace=True)

snow_df = snow_df.iloc[::-1]
print(snow_df)

visibility_df = weather_df['visibility']
for i in range(len(visibility_df)):
    if visibility_df[i] in range(1000):
        visibility_df[i] = 1
    if visibility_df[i] in range(1001, 4000):
        visibility_df[i] = 2
    if visibility_df[i] in range(4001, 10000):
        visibility_df[i] = 3
    if visibility_df[i] > 10000:
        visibility_df[i] = 4
print(visibility_df)
weather_df['visibility'] = visibility_df

snow_depth_df = weather_df['snow_depth']
for i in range(len(snow_depth_df)):
    snow_depth_df[i]*= 100
print(snow_depth_df)
weather_df['snow_depth'] = snow_depth_df

print(weather_df)

x_index = list(range(0,len(weather_df['snow_depth'])))
print(x_index)
print(weather_df['temperature_2m'])
print(snow_df['ta'].iloc[0:])
plt.plot(x_index, weather_df['weathercode'], label='first')
plt.plot(x_index, snow_df['ww'].iloc[0:], label='second')
plt.legend()
plt.show()