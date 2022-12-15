import gradio as gr

import hopsworks
import pandas as pd
import joblib

project = hopsworks.login(project="finetune")

fs = project.get_feature_store()

predictions = fs.get_feature_group(name="snow_predictions",version=1)
print(predictions) 
#da capire, prediction è un fg non un dataframe
#probabile va caricato un grafico/immagini sul file directory di hopsworks e poi fatto il retrival

with gr.Blocks() as demo:
    with gr.Row():
        gr.Text("Snow Level Predictor")
        gr.DataFrame(predictions)

demo.launch()