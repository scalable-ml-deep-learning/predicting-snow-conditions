import os
import modal
from sklearn import datasets
import xgboost as xgb 

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks==3.0.4", "seaborn", "joblib", "scikit-learn"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("SNOW_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib

    from sklearn import datasets
    import xgboost as xgb   
    import numpy as np
    from sklearn.metrics import precision_score, recall_score, accuracy_score

    # You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
    project = hopsworks.login(project="finetune")
    # fs is a reference to the Hopsworks Feature Store
    fs = project.get_feature_store()

    # The feature view is the input set of features for your model. The features can come from different feature groups.    
    # You can select features from different feature groups and join them together to create a feature view
    feature_view = fs.get_feature_view(name="snow_weather_data", version=1)

    # You can read training data, randomly split into train/test sets of features (X) and labels (y)        
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

    X_train = X_train.sort_values(by=["time"], ascending=[False]).reset_index(drop=True)
    X_test = X_test.sort_values(by=["time"], ascending=[False]).reset_index(drop=True)
    
    X_train = X_train.drop(columns=["time"]).fillna(0)
    X_test = X_test.drop(columns=["time"]).fillna(0)

    #y = X.pop("aqi_next_day")
    #gb = GradientBoostingRegressor()
    #gb.fit(X, y)

    D_train = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    D_test = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

    steps = 1 

    model = xgb.train(param, D_train, steps)

    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
    print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
    print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))
    
    
    # Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
    #model = KNeighborsClassifier(n_neighbors=2)
    #model.fit(X_train, y_train.values.ravel())

    # Evaluate model performance using the features from the test set (X_test)
    #y_pred = model.predict(X_test)

    # Compare predictions (y_pred) with the labels in the test set (y_test)
    #metrics = classification_report(y_test, y_pred, output_dict=True)
    #results = confusion_matrix(y_test, y_pred)

    # Create the confusion matrix as a figure, we will later store it as a PNG image file
    #df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                         #['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
    #cm = sns.heatmap(df_cm, annot=True)
    #fig = cm.get_figure()

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    #mr = project.get_model_registry()
    
    # The contents of the 'iris_model' directory will be saved to the model registry. Create the dir, first.
    #model_dir="iris_model"
    #if os.path.isdir(model_dir) == False:
        #os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    #joblib.dump(model, model_dir + "/iris_model.pkl")
    #fig.savefig(model_dir + "/confusion_matrix.png")    


    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    #input_schema = Schema(X_train)
    #output_schema = Schema(y_train)
    #model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    #iris_model = mr.python.create_model(
    #    name="iris_modal", 
    #    metrics={"accuracy" : metrics['accuracy']},
    #    model_schema=model_schema,
    #    description="Iris Flower Predictor"
    #)
    
    # Upload the model to the model registry, including all files in 'model_dir'
    #iris_model.save(model_dir)
    
if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()