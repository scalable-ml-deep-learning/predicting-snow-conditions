import gradio as gr
import numpy as np
from PIL import Image
import requests
import matplotlib.pyplot as plt
import hopsworks
import joblib


def get_prediction():
    '''
    Get the latest prediction from Hopsworks.
    Returns the prediction.
    '''
    project = hopsworks.login()
    fs = project.get_feature_store()

    prediction_fg = fs.get_feature_group(name="snow_predictions", version=1)
    prediction = prediction_fg.read()
    
    return prediction
    

def plot_prediction(prediction):
    '''
    Get the prediction and create a plot showing the snow depth
    for the following days.
    '''
    '''
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    print(res)
    passenger_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/serverless-ml-for-titanic/feature-brando/img/" + str(res[0]) + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)   
    '''
    plt.plot(prediction["time"], prediction["snow_level_prediction"])
    plt.savefig("plot.png")
             
    return
    
def get_image_snow_depth(snow_depth):
    '''
    Get a snow depth level and return an image 
    indicating good level / bad level with emoji
    '''
    '''
    passenger_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/serverless-ml-for-titanic/feature-brando/img/" + str(res[0]) + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)
    '''
    return image
    
with gr.Blocks() as demo:
  with gr.Row():
    pred = get_prediction()
    print("Prediction:\n", pred)
    print("Time:\n", pred["time"])
    print("Snow:\n", pred["snow_level_prediction"])
    plot_prediction(pred)
    plot_pred = gr.Image("plot.png") # plotted graph
  with gr.Row():
    emojis = gr.Image("emojis.jpeg") # table of emojis
  with gr.Row():  
    btn = gr.Button("New prediction").style(full_width=True)

demo.launch()
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
