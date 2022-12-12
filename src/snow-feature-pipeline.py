import os
import hopsworks
import pandas as pd

project = hopsworks.login(project="finetune")
fs = project.get_feature_store()

snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")

snow_df = snow_df.loc[snow_df['codStaz'] == "31RO"] # Select Passo Rolle
snow_df.drop(['codStaz', 'oraDB','ww', 'n', 'v', 'vq1', 'vq2', 'ta', 'tmin', 'tmax', 'hn', 'fi', 't10', 't30', 'pr', 'cs', 's', 'b'], axis='columns', inplace=True)
snow_df.dropna(inplace=True)
print(snow_df)

snow_fg = fs.get_or_create_feature_group(
    name="snow_level",
    version=1,
    primary_key=["dataMis"], 
    description="Snow level dataset")
snow_fg.insert(snow_df, write_options={"wait_for_job" : False})