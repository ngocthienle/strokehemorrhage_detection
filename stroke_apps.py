# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:57:43 2021
Stroke Image Classification project - Chula
Stroke LVO AI/app Chula
@author: thienle
Date: July 14, 2021
"""
# Import libraries
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np

import os, urllib, cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input

# NAME OF PAGE
logo_image = Image.open('logo.png')
st.set_page_config(page_title = 'StrokeApps', page_icon=logo_image)

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("aimodels/multi_stroke_newversion.hdf5")
    return model

with st.spinner('Loading Model Into Memory...'):
    model = load_model()

# Create web-app title
st.title("""Stroke LVO Web Application Chulalongkorn University""")

# Create header explaination
st.write("""
    This web-apps predicts the CT-brain image is normal or abnormal (with signs of stroke).
    The backend system is a trained AI model.
    """)
    
st.subheader('Example input CT-brain images')
image = Image.open('example_pics.jpg')
st.image(image, caption='Example CT-brain images. Source of these images from CQ500 publish dataset (http://headctstudy.qure.ai/dataset)')

st.subheader('Choose a CT-brain image and get the output prediction')
uploaded_file = st.file_uploader("Upload your input jpeg file", type=["jpg"])

map_dict = {6: 'background.',
            5: 'abnormal at center brain layer', # hge
            4: 'abnormal at eyeball brain layer.', #hge_eyeball
            3: 'abnormal at top brain layer.', #hge_top
            2: 'normal at center brain layer.',#nonehge
            1: 'normal at eyeball brain layer.',#nonehge_eyeball
            0: 'normal at top brain layer.', #nonehge_top
            }

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]

    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(img_reshape).argmax()
        st.title("The CT-brain image is {}".format(map_dict [prediction]))
