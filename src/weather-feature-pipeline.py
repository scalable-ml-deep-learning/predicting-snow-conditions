import os
import pandas as pd
import matplotlib.pyplot as plt

# take weather dataset
# hourly at midnight
weather_df = pd.read_json("https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&hourly=temperature_2m,snow_depth,weathercode,visibility&models=best_match&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-11")
# weather dataset daily
#weather_df = pd.read_json("https://api.open-meteo.com/v1/forecast?latitude=46.2979&longitude=11.7871&models=best_match&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto&start_date=2022-12-04&end_date=2022-12-12")
# take snow dataset
snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")

#####################################
# DATA WRANGLING ON WEATHER DATASET #
#####################################

# Drop unwanted features
weather_df.drop(['latitude', 'longitude','hourly_units', 'generationtime_ms', 'utc_offset_seconds', 'timezone', 'timezone_abbreviation', 'elevation', 'current_weather'], axis='columns', inplace=True)
weather_df.dropna(inplace=True)

# take values for the midnight of every day in the dataframe
weather_df = weather_df['hourly']
for key in weather_df.keys():
    df_ = []
    for i in range(0, len(weather_df[key]), 24):
        df_.append(weather_df[key][i])
    weather_df[key] = df_

print("Weather dataframe: ", weather_df)

# change visibility from meters to a range 1-4
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

weather_df['visibility'] = visibility_df

# convert snow depth from meters to centimeters
snow_depth_df = weather_df['snow_depth']

for i in range(len(snow_depth_df)):
    snow_depth_df[i] *= 100
    
print(snow_depth_df)

weather_df['snow_depth'] = snow_depth_df

##################################
# DATA WRANGLING ON SNOW DATASET #
##################################

# Select Passo Rolle 31RO
# Select Passo del Tonale 25TO
snow_df = snow_df.loc[snow_df['codStaz'] == "25TO"] 
# drop unwanted features
snow_df.drop(['codStaz', 'oraDB', 'vq1', 'vq2', 'n', 'tmin', 'tmax', 'hn', 'fi', 'pr', 't10', 't30', 'cs', 's', 'b'], axis='columns', inplace=True)
snow_df.dropna(inplace=True)

snow_df = snow_df.iloc[::-1]
print("Snow dataframe:", snow_df)

###################################
# INSIGHTS ON DATA AND COMPARISON #
###################################

x_index = list(range(0,8))

# temperature
print("Temperature from weather_df:\n", weather_df['temperature_2m'])
print("Temperature from snow_df:\n", snow_df['ta'].values)
# visibility
print("Visibility from weather_df:\n", weather_df['visibility'])
print("Visibility from snow_df:\n", snow_df['v'].values)
# weather code
print("Weathercode from weather_df:\n", weather_df['weathercode'])
print("Weathercode from snow_df:\n", snow_df['ww'].values)

'''
plt.plot(x_index, weather_df['weathercode'], label='first')
plt.plot(x_index, snow_df['ww'].iloc[0:], label='second')
plt.legend()
plt.show()
'''















