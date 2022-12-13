import os
import modal
import time
import datetime

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("snow_data_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "lxml", "python-time", "DateTime"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd
    import time
    import datetime

    project = hopsworks.login(project="finetune")
    fs = project.get_feature_store()

    snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")
    snow_df = snow_df.loc[snow_df['codStaz'] == "31RO"]
    snow_df.drop(['codStaz', 'oraDB','ww', 'v', 'vq1', 'vq2', 'n', 'ta', 'tmin', 'tmax', 'hn', 'fi', 't10', 't30', 'pr', 'cs', 's', 'b'], axis='columns', inplace=True)
    snow_df.dropna(inplace=True)
    snow_df = snow_df.iloc[:1]
    #print(snow_df)
    
    for value in snow_df['dataMis'].values:
        value_date = value.split(" ")[0]
        value_date = value_date.split("/")
        day = value_date[0]
        month = value_date[1]
        year = value_date[2]
        value_date = f'{year}-{month}-{day}'
        #print(value_date)
        snow_df.replace(to_replace=value, value=value_date, inplace = True)

    snow_df.rename(columns = {'dataMis':'time'}, inplace = True)
    print(snow_df)

    snow_fg = fs.get_feature_group(name="snow_data",version=1)
    snow_fg.insert(snow_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("snow_data_daily")
        with stub.run():
            f()
