import os
import modal
import time
import datetime

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("snow_level_data_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "lxml"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import lxml
    import pandas as pd

    project = hopsworks.login(project="finetune")
    fs = project.get_feature_store()

    snow_df = pd.read_xml("http://dati.meteotrentino.it/service.asmx/tuttiUltimiRilieviNeve")
    snow_df = snow_df.loc[snow_df['codStaz'] == "31RO"]
    snow_df.drop(['codStaz', 'oraDB','ww', 'v', 'vq1', 'vq2', 'n', 'ta', 'tmin', 'tmax', 'hn', 'fi', 't10', 't30', 'pr', 'cs', 's', 'b'], axis='columns', inplace=True)
    snow_df.dropna(inplace=True)
    snow_df = snow_df.iloc[:1]
    #print(snow_df)
    
    for value in  snow_df['dataMis'].values:
        value_date = value.split(" ")[0]
        #print("Value: ", value_date)
        new_value = time.mktime(datetime.datetime.strptime(value_date, "%d/%m/%Y").timetuple())
        #print("New value: ", new_value)
        snow_df.replace(to_replace=value, value=new_value, inplace = True)

    print(snow_df)
    snow_fg = fs.get_feature_group(name="snow_level",version=1)
    snow_fg.insert(snow_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("snow_level_data_daily")
        with stub.run():
            f()
