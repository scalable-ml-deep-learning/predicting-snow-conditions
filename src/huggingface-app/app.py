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

def sort_by_time(df):
    '''
    Sort dataframe by time
    '''
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by='time', ascending=True, inplace=True)
    return df

def get_actual_snow():
    '''
    Get from Hopsworks the actual snow level day by day
    since the beginning of the collection.
    '''
    snow_data_fg = fs.get_feature_group(name="snow_data")
    snow_data = snow_data_fg.read()
    # sort by time
    snow_data = sort_by_time(snow_data)
    
    return snow_data
    
def plot_actual_snow(actual_snow):
    '''
    Plot the actual snow level vs time.
    '''
    plt.plot(actual_snow['time'], actual_snow['hs'], label='actual snow', linestyle=":", color='blue')
    return

def get_snow_prediction():
    '''
    Get the latest prediction from Hopsworks.
    Returns the prediction.
    '''
    prediction_fg = fs.get_feature_group(name="snow_predictions", version=1)
    prediction = prediction_fg.read()
    # sort by time
    prediction = sort_by_time(prediction)
    
    return prediction
    
#def plot_today_date_bar():
#    '''
#    Get today's date. Plot a vertical red line on today's date.
#    '''
#    return
    

def plot_prediction():
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
    actual_snow = get_actual_snow()
    plot_actual_snow(actual_snow)
    predicted_snow = get_snow_prediction()
    plt.plot(predicted_snow['time'], predicted_snow['snow_level_prediction'], label='predicted snow', linestyle="--", color="red")
    #plot_today_date_bar()
    plt.legend()
    #plt.show()
    plt.savefig("plot.png")
             
    return
    
def get_image_snow_depth(emojis):
    '''
    Get a snow depth level and return an image 
    indicating good level / bad level with emoji
    '''
    '''
    passenger_url = "https://raw.githubusercontent.com/scalable-ml-deep-learning/serverless-ml-for-titanic/feature-brando/img/" + str(res[0]) + ".png"
    img = Image.open(requests.get(passenger_url, stream=True).raw)
    '''
    return np.fliplr(emojis)

# Uncomment for gradio interface locally on the browser

with gr.Blocks() as demo:
    with gr.Row():
      plot_prediction()
      plot_pred = gr.Image("plot.png", label="Predicted snow hight", ) # plotted graph
    with gr.Row():
      image_input = gr.Image("emojis.jpeg", visible=True) # table of emojis
      image_output = gr.Image()
    with gr.Row():  
      btn = gr.Button("New prediction").style(full_width=True)
    
    btn.click(get_image_snow_depth, inputs=image_input, outputs=image_output)

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
