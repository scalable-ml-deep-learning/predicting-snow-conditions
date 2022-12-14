# predicting-snow-conditions

## Useful resources:
[Hopsworks](https://www.hopsworks.ai/): can have access with fine-tune account and our personal accounts, project finetune.

[Modal](https://modal.com/): have access with personal accounts, on Organization workspace scalable-ml-deep-learning

[Open-weather data](https://open-meteo.com/en/docs#latitude=46.2979&longitude=11.7871&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,precipitation,rain,showers,snowfall,snow_depth,freezinglevel_height,visibility&models=best_match&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,showers_sum,snowfall_sum,precipitation_hours&current_weather=true&timezone=auto&past_days=61): download data of eight days and update daily

[Snow data Trentino](https://dati.trentino.it/dataset/dati-recenti-dei-campi-neve/resource/0bbde12d-348d-43ea-8a30-078d59df5188): download data of eight days and update daily. [Reference table](http://content.meteotrentino.it/neve-ghiacci/Husky/mod1/legenda-mod1.pdf) for atmospheric values.

[Passo Rolle position](https://goo.gl/maps/G3Qw8WNvZ19ojKEK7) 2012m altitude, 31RO code

XGBoost model: [tutorial on timeseries for weather forecasting](https://www.kaggle.com/code/robikscube/tutorial-time-series-forecasting-with-xgboost), [XGBoost API Reference Python](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn). Difference between [.train()](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training) and [.fit()](https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn) can be found [here](https://stackoverflow.com/questions/47152610/what-is-the-difference-between-xgb-train-and-xgb-xgbregressor-or-xgb-xgbclassif). Both are to train a model. In the case of XGBoost, one can use xgboost.train() to train a model on the passed data. Or use the Scikit-Learn API, where different XGBoost models can be applied, for example, xgboost.XGBRegressor, xgboost.XGBClassifier, xgboost.XGBRanker, and many others. These last are Scikit-Learn wrappers, meaning that are fine-tuned models that perform feature selection to reduce the number of predictors used to train the model. More information can be foound in [this](https://towardsdatascience.com/feature-selection-for-machine-learning-in-python-wrapper-methods-2b5e27d2db31) blog post.

GCP Deep Learning: [tutorial](https://medium.com/google-cloud/how-to-run-deep-learning-models-on-google-cloud-platform-in-6-steps-4950a57acfa5)

## Useful commands

`modal token set --token-id <your_token-id> --token-secret <your_token-secret> --env=scalable_ml-deep-learning`

`modal env list`
`modal env activate scalable_ml-deep-learning`
