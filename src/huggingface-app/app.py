import gradio as gr
from PIL import Image
import hopsworks
import joblib

# Connect to Hopsworks
project = hopsworks.login(project="finetune")
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

###########################################
# DOWNLOAD AND OPEN IMAGES FROM HOPSWORKS #
###########################################
def show_reloaded_images():
    '''
    Show new images.
    '''
    # download emoticons
    for day in range(1,7):
      img = f'Resources/img_prediction/{day}.png'
      dataset_api.download(img, overwrite=True)
    # download snow prediction forecast
    dataset_api.download("Resources/img_prediction/plot.png", overwrite=True)
    # output images
    plot_pred = Image.open("plot.png")
    img1 = Image.open("1.png")
    img2 = Image.open("2.png")
    img3 = Image.open("3.png")
    img4 = Image.open("4.png")
    img5 = Image.open("5.png")
    img6 = Image.open("6.png")
    output = [plot_pred, img1, img2, img3, img4, img5, img6]
  
    return output

######################################
# DOWNLOAD AND SHOW ACCURACY HISTORY #
######################################
def show_history():
    '''
    Get history of predictions.
    '''
    dataset_api.download("Resources/img_prediction/plot_history.png", overwrite=True)
    plot_hist = Image.open("plot_history.png")
    
    return plot_hist

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Snow prediction"):
            with gr.Row():  
                btn = gr.Button("New prediction").style(full_width=True)
            with gr.Row():
                plot_pred = gr.Image(label="Predicted snow height").style(height=400) # plotted graph
            with gr.Row(equal_width=True):
                #input_img1 = gr.Image("1.png", elem_id="Day 1")
                img1 = gr.Image(label="Tomorrow")
                img2 = gr.Image(label="In 2 days")
                img3 = gr.Image(label="In 3 days")
                img4 = gr.Image(label="In 4 days")
                img5 = gr.Image(label="In 5 days")
                img6 = gr.Image(label="In 6 days")
            with gr.Row():
                img_legend = gr.Image("emoji-legend.png", label="Legend").style(height=300)
              
        with gr.TabItem("Accuracy of past 10 days"):
            with gr.Row():
                btn2 = gr.Button("Get history").style(full_width=True)
            with gr.Row():
                pred_hist = gr.Image(label="Past 10 days of predictions").style(height=500)
      
    btn.click(show_reloaded_images, 
    inputs=None, 
    outputs=[plot_pred, img1, img2, img3, img4, img5, img6])
    
    btn2.click(show_history,
    inputs=None,
    outputs=pred_hist)

demo.launch()
