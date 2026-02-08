import streamlit as st
import json
import os
import numpy as np
import tempfile
from tensorflow.keras.utils import img_to_array
import tensorflow as tf
from tensorflow import keras
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))
model = keras.models.load_model(os.path.join(dir_path, "models/model_v1.keras"))
with open(os.path.join(dir_path, "classes_dict.json"), "r") as file:
    class_names = json.load(file)

def predict(model, img_path):
    img = Image.open(img_path)
    img_array = img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)

    predicted_class = class_names["class_names"][np.argmax(predictions[0])]
    confidence = round(100 * (np.argmax(predictions[0])), 2)
    return predicted_class, confidence


background_image ="""
<style>
[data-testid="stAppViewContainer"]{
  background-image: url("https://img.freepik.com/free-photo/sunny-bokeh-garden-leaf-bright_1253-470.jpg?semt=ais_hybrid&w=740");
  background-size: cover;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

st.title("Deep Learning Model: Plant Disease Classifier")
uploaded_image = st.file_uploader( "Drag and drop an Image to Identify:", type=['jpg', 'png'],key='file_uploader')
if uploaded_image:
    st.image(uploaded_image)
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, uploaded_image.name)
    with open(img_path, "wb") as f:
            f.write(uploaded_image.getvalue())
            
    predicted_class, confidence = predict(model, img_path)
    
    st.success(f"Predicted Disease: {predicted_class}")
    st.success(f"Confidence: {confidence}")

st.write("Or")
camera_image = st.camera_input("Take a picture")
if camera_image:
    temp_dir = tempfile.mkdtemp()
    img_path = os.path.join(temp_dir, camera_image.name)
    with open(img_path, "wb") as f:
            f.write(camera_image.getvalue())
            
    predicted_class, confidence = predict(model, img_path)
    
    st.success(f"Predicted Disease: {predicted_class}")
    st.success(f"Confidence: {confidence}")

    
    
        


        

    
