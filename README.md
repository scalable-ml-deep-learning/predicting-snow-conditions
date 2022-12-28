# Predicting Snow Conditions


Contributors:
<a href="https://github.com/Bralli99">Brando Chiminelli</a>, 
<a href="https://github.com/boyscout99">Tommaso Praturlon</a>

Course: <a href="https://id2223kth.github.io/">Scalable Machine Learning and Deep Learning</a>, at <a href="https://www.kth.se/en">KTH Royal Institute of Technology</a>

## About

The project is about building an automated and scalable system that predicts the snow conditions for [Passo Rolle](https://goo.gl/maps/G3Qw8WNvZ19ojKEK7) (ski area in Trento, Italy) for the upcoming week, so that a skier that the plans to go skiing on the next weekend, can look in advance if the conditions are good. 
In particular, the system predicts the ground snow level, and give advices to skiers based on what it predicted, the forecasted visibility and wind condition, three important factors for fun and safe skiing.

## Project's Architecture
An overview of the project's architecture is present below, next all the different sections are explained.
<img src="./src/images/pipelines_diagram.png" alt="pipelines diagram" width="100%"/> 

### Data Model
The first thing to look for when dealing with a prediction service is to look for good quality data that is updated regualrly. In this project we are dealing with data that is updated everyday and the source has been verified.

To predict the snow level we need to think which factors contribute the most to the increase/decrease of the snow level. The one used in this project are taken from [Open-Meteo](https://open-meteo.com/) and are the following:
* Weathercode: Weather condition as a numeric code. Follow WMO weather interpretation codes.
* Temperature: Temperature taken distinctly as max/min of the day.
* Precipitation: Sum of daily precipitation (divided into rain, showers and snowfall).
* Precipitation hours: The number of hours with precipitation.

The historical data for the snow level is provided by [Open Data Trentino](https://dati.trentino.it/), unfortunately it gives only the last eight days of data. Nevertheless, the system has been collecting data everyday since the 6th of December, 2022, so the dataset keeps growing and the predictions keep getting better.

By looking at the project's architecture diagram it is clear that the system does not only collect static historical data, but deals with new data ingestions every day, resulting in updated and useful future predictions.

### Feature Pipeline
Data is ingested into [Hopsworks](https://www.hopsworks.ai/) Feature Store with the help of two main pipelines, one for historical data and one for fresh new data. The first one runs on demand as it takes a batch of historical data, does some data cleaning, and writes the final result to a feature group (one called `weather_data` and the other `snow_data`).

The second pipeline, takes the latest weather and snow data, does some data cleaning and adds a row on both the two feature groups. This last pipeline runs every day as the data coming from the two data sources is updated once per day. To make it running every day the pipeline runs on a container in the cloud, completely handled by [Modal](https://modal.com/).

Lastly, a Feature View is created by joining the two feature groups created before, `weather_data` and `snow_data`. 
A Feature View is a logical view over the data present on the join of the two feature groups, it is an easy way to redefine data types and specify a label, later needed for the training pipeline.

### Training Pipeline
The training pipeline runs everyday on Modal so that the model can improve in time, since it's trained every day with more data. The training starts by retrieving the data from the feature view with a specific function `feature_view.training_data()` and then performs three operations on the data:
1. Order the data by time (ascending order);
2. Removes the time feature as we don't need that for training;
3. Splits the data into train (the first 80%) and test (the last 20%) set.

The model used for training is a gradient boosted decision tree, specifically the XGBoost python model implementation with the scikit-learn API via the XGBRegressor wrapper class.
After trying with different configurations, the parameters that work the best and so selected for the training are:
*`'max_depth': 3` = Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
*`'objective': 'reg:squarederror'` = The loss function 'reg:squarederror' is suitable for regression predictive modeling problems.


### Inference Pipeline
The inference pipeline is responsible for making the predictions by using the best model so far trained (based on its mean squared error). The inference pipeline writes all its predictions on the 'snow_predictions' feature group and runs on Modal everyday, so that the predictions improve as we get closer to the ski weekend (as the weather forecast will also improve). Lastly, the inference pipeline generates a plot for historical values of snow levels and the corresponding predicted values. It also generate an advice for the skier in the form of an emoji, which is, together with the plot, saved to the Hopsworks File System.

### User Interface
The User Interface simply shows the latest predictions and the historical values for snow level in Passo Rolle and gives advices to skiers for the upcoming weekend. It shows a plot and some images that are present and updated daily on Hopsworks File System.

## Useful resources:
[Open-weather data](https://open-meteo.com/en/docs#latitude=46.2979&longitude=11.7871&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,showers,snowfall,snow_depth,freezinglevel_height,visibility&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto&past_days=61): download data of eight days and update daily

[Snow data Trentino](https://dati.trentino.it/dataset/dati-recenti-dei-campi-neve/resource/0bbde12d-348d-43ea-8a30-078d59df5188): download data of eight days and update daily. [Reference table](http://content.meteotrentino.it/neve-ghiacci/Husky/mod1/legenda-mod1.pdf) for atmospheric values.

[Passo Rolle position](https://goo.gl/maps/G3Qw8WNvZ19ojKEK7) 2012m altitude, 31RO code

XGBoost model: [tutorial on timeseries for weather forecasting](https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost), [XGBoost API Reference Python](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn). Difference between [.train()](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training) and [.fit()](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) can be found [here](https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif). Both are to train a model. In the case of XGBoost, one can use xgboost.train() to train a model on the passed data. Or use the Scikit-Learn API, where different XGBoost models can be applied, for example, xgboost.XGBRegressor, xgboost.XGBClassifier, xgboost.XGBRanker, and many others. These last are Scikit-Learn wrappers, meaning that are fine-tuned models that perform feature selection to reduce the number of predictors used to train the model. More information can be found in [this](https://towardsdatascience.com/feature-selection-for-machine-learning-in-python-wrapper-methods-2b5e27d2db31) blog post.

99\% of XGBoost models only need to tune: objective function, number of leaves or max depth, feature fraction, bagging fraction, min child fraction.

GCP Deep Learning: [tutorial](https://medium.com/google-cloud/how-to-run-deep-learning-models-on-google-cloud-platform-in-6-steps-4950a57acfa5)

## Useful commands

`modal token set --token-id <your_token-id> --token-secret <your_token-secret> --env=scalable_ml-deep-learning`

`modal env list`
`modal env activate scalable_ml-deep-learning`

## Built With

* [Hopsworks](https://www.hopsworks.ai/)
* [Modal](https://modal.com/)
