import gradio as gr
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import hopsworks
import joblib
import pandas as pd

# Connect to Hopsworks
project = hopsworks.login(project="finetune")
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()
'''
for day in range(1,7):
    img = f'Resources/img_prediction/{day}.png'
    dataset_api.download(img, overwrite=True)
    
dataset_api.download("Resources/img_prediction/plot.png", overwrite=True)
'''
 
def reload_images():
    '''
    Get a snow depth level and return an image 
    indicating good level / bad level with emoji
    '''
    for day in range(1,7):
      img = f'Resources/img_prediction/{day}.png'
      dataset_api.download(img, overwrite=True)
    
    dataset_api.download("Resources/img_prediction/plot.png", overwrite=True)
    print("Clicked")
    # refresh page
    return
    
def show_reloaded_images():
    '''
    Show new images.
    '''
    #print("img:\n", plot_pred)
    #print("Path: ", plot_pred.name)
    #resized_img = plot_pred.clear()
    plot_pred = gr.Image("emoji.jpeg", label="Predicted snow height")
    #img.clear()
    print("img:\n", plot_pred)
    print("Changed.")
    
    return

# Uncomment for gradio interface locally on the browser

with gr.Blocks() as demo:
    with gr.Row():
      plot_pred = gr.Image("plot.png", type="numpy", label="Predicted snow height")#.style(height=500) # plotted graph
      print("Plot_pred:\n", plot_pred)
    with gr.Row():
      input_img1 = gr.Image("1.png", elem_id="Day 1")
      input_img2 = gr.Image("2.png", elem_id="Day 2")
      #gr.Label("Today's Predicted Image")
      input_img3 = gr.Image("3.png", elem_id="Day 3")
      input_img4 = gr.Image("4.png", elem_id="Day 4")
      input_img5 = gr.Image("5.png", elem_id="Day 5")
      input_img6 = gr.Image("6.png", elem_id="Day 6")
      
    with gr.Row():  
      btn = gr.Button("New prediction").style(full_width=True)
      print("Button:\n", btn)
      
    print("again Plot_pred:\n", plot_pred)
    btn.click(show_reloaded_images, inputs=None, outputs=None)

demo.launch()

# Previous demo interface
'''      
demo = gr.Interface(
    fn=passenger,
    title="Snow depth level prediction for Passo Rolle (TN), Italy",
    description="Just having a forecast of the snow is not enough! From today skiers and snowboarders can actually have a forecast of the snow level!",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1.0, label="Passenger class"),
        gr.inputs.Number(default=1.0, label="Passenger sex (1:male, 0:female)"),
        gr.inputs.Number(default=1.0, label="Passenger age"),
        gr.inputs.Number(default=1.0, label="Passenger fare")
        ],
    outputs=gr.Image(type="pil"))

demo.launch()
'''
