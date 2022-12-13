import os
import hopsworks
import pandas as pd
import time
import datetime

project = hopsworks.login(project="finetune")
fs = project.get_feature_store()

snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")

snow_df = snow_df.loc[snow_df['codStaz'] == "31RO"] # Select Passo Rolle
snow_df.drop(['codStaz', 'oraDB','ww', 'n', 'v', 'vq1', 'vq2', 'ta', 'tmin', 'tmax', 'hn', 'fi', 't10', 't30', 'pr', 'cs', 's', 'b'], axis='columns', inplace=True)
snow_df.dropna(inplace=True)
#print(snow_df)

#################
# DATA CLEANING #
#################
# convert dates to yyyy-mm-dd to prepare join
for value in snow_df['dataMis'].values:
  value_date = value.split(" ")[0]
  value_date = value_date.split("/")
  day = value_date[0]
  month = value_date[1]
  year = value_date[2]
  value_date = f'{year}-{month}-{day}'
  #print(value_date)
  snow_df.replace(to_replace=value, value=value_date, inplace = True)

print(snow_df)

snow_fg = fs.get_or_create_feature_group(
    name="snow_data",
    version=1,
    primary_key=["dataMis"], 
    description="Snow level dataset")
snow_fg.insert(snow_df, write_options={"wait_for_job" : False})
