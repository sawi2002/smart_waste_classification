import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

model = tf.keras.models.load_model("waste_model.h5", compile=False)

with open("class_names.json") as f:
    class_names = json.load(f)

st.title("♻️ Waste Classification App")

upload = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if upload is not None:
    img = Image.open(upload)
    st.image(img)

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]

    st.success(class_names[np.argmax(pred)])
