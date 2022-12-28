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
    for day in range(1,7):
      img = f'Resources/img_prediction/{day}.png'
      dataset_api.download(img, overwrite=True)
    
    dataset_api.download("Resources/img_prediction/plot.png", overwrite=True)
    plot_pred = Image.open("plot.png")
    img1 = Image.open("1.png")
    img2 = Image.open("2.png")
    img3 = Image.open("3.png")
    img4 = Image.open("4.png")
    img5 = Image.open("5.png")
    img6 = Image.open("6.png")
    output = [plot_pred, img1, img2, img3, img4, img5, img6]
  
    return output

# Uncomment for gradio interface locally on the browser

with gr.Blocks() as demo:
    with gr.Row():
      plot_pred = gr.Image(label="Predicted snow height").style(height=500) # plotted graph
      print("Plot_pred:\n", plot_pred)
    with gr.Row():
      #input_img1 = gr.Image("1.png", elem_id="Day 1")
      img1 = gr.Image()
      img2 = gr.Image()
      #gr.Label("Today's Predicted Image")
      img3 = gr.Image()
      img4 = gr.Image()
      img5 = gr.Image()
      img6 = gr.Image()
    with gr.Row():  
      btn = gr.Button("New prediction").style(full_width=True)
      print("Button:\n", btn)
      
    btn.click(show_reloaded_images, 
    inputs=None, 
    outputs=[plot_pred, img1, img2, img3, img4, img5, img6])
    print("again Plot_pred:\n", plot_pred)
    #demo.load()

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
