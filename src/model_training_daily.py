import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub("snow_model_training_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4", "joblib", "scikit-learn", "xgboost"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    import xgboost as xgb   
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, accuracy_score
    from sklearn.metrics import mean_squared_error as MSE

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login(project="finetune")
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.    
    # You can select features from different feature groups and join them together to create a feature view
    feature_view = fs.get_feature_view(name="snow_weather_data", version=1)

    # You can read training data, randomly split into train/test sets of features (X) and labels (y)        
    #X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)
    X_train, y_train = feature_view.training_data()

    print("X_train:", X_train)
    print("y_train", y_train)

    X_train = X_train.sort_values(by=["time"], ascending=[True])
    
    X_train = X_train.drop(columns=["time"]).fillna(0)

    index_list = X_train.index.tolist()
    print(index_list)
    y_train = y_train.reindex(index_list)
    #y_train.loc[index_list]

    train_perc = int(len(X_train) // 1.25) #take 80perc
    X_test = X_train.iloc[train_perc:]
    y_test = y_train.iloc[train_perc:]
    X_train = X_train.iloc[:train_perc]
    y_train = y_train.iloc[:train_perc]

    kwargs = {
    'max_depth': 3,  
    'objective': 'reg:squarederror',  
    }
    # XGBoost regression: 
    # Parameters: 
    # n_estimators  "Number of gradient boosted trees. Equivalent to number of boosting rounds."
    # learning_rate "Boosting learning rate (also known as “eta”)"
    # max_depth     "Maximum depth of a tree. Increasing this value will make 
    #                the model more complex and more likely to overfit." 
    # 99% of XGBoost models only need to tune: objective function, number of leaves or max depth, 
    # feature fraction, bagging fraction, min child fraction.

    xgb_r = xgb.XGBRegressor(**kwargs)
    
    print("X_train:", X_train)
    print("y_train", y_train)
    print("X_test:", X_test)
    print("y_test", y_test)

    # Fitting the model
    xgb_r.fit(X_train, y_train)

    pred = xgb_r.predict(X_test)
 
    # RMSE Computation
    rmse = np.sqrt(MSE(y_test, pred))
    print("RMSE : % f" %(rmse))
    print("prediction:", pred)
    print("real values:", y_test)

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()
    
    # The contents of the 'snow_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="snow_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(xgb_r, model_dir + "/snow_model.pkl")

    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    snow_model = mr.python.create_model(
        name="snow_model", 
        metrics={"mean squared error" : rmse},
        model_schema=model_schema,
        description="Snow Level Predictor"
    )
    
    # Upload the model to the model registry, including all files in 'model_dir'
    snow_model.save(model_dir)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("snow_model_training_daily")
        with stub.run():
            f()
