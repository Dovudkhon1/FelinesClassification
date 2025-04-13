import streamlit as st
import plotly.express as px
import pathlib
from fastai.vision.all import *

#title
st.title("Model to classify types of felines between Leopards, Tigers, Lions and Cats")

#upload file
file = st.file_uploader("Upload an image", type=['jpeg', 'png', 'gif',])

if file:
    st.image(file)
    img = PILImage.create(file)
    #model
    model = load_learner('FelinesModel.pkl')
    pred, pred_idx, probs = model.predict(img)

    st.success(f"Prediction: {pred}")
    st.info(f"Accuracy: {probs[pred_idx]*100:.1f}%")

    #graph
    fig = px.bar(x=probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)
