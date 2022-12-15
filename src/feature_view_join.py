import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login(project="finetune")
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.    
    # You can select features from different feature groups and join them together to create a feature view
    try:
        feature_view = fs.get_feature_view(name="snow_weather_data", version=1)
    except:
        snow_fg = fs.get_feature_group(name="snow_data", version=1)
        weather_fg = fs.get_feature_group(name="weather_data", version=1)
        query = snow_fg.select_all().join(weather_fg.select_all())
        feature_view = fs.create_feature_view(name="snow_weather_data",
                                          version=1,
                                          description="Snow and Weather Data",
                                          labels=["hs"],
                                          query=query)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()