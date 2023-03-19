import streamlit as st
import pandas as pd
import altair as alt


@st.cache_data
def get_labels():
    return pd.read_csv("train/labels.csv")


@st.cache_data
def plot_classes():
    data = df_labels["breed"].value_counts().reset_index(name="count")
    chart = (
        alt.Chart(data).mark_bar().encode(x="count", y="index").properties(width=700)
    )
    st.altair_chart(chart)


"# Model's Learning Journey 🤖"

"## Introduction"

"The goal here was to create a machine learning model that could identify the breed of a dog in a given image."
"Convolution Neural Networks are really good at working with images. Therefore we could leverage on existing models trained on identifying general objects in images and fine tune it to our specific niche, dogs 🐕."
"I started with [MobileNetV2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5) and later on switched to [EfficientNet-B0](https://tfhub.dev/tensorflow/efficientnet/b0/classification/1)."

"## Data"

"The data was sourced from [Kaggle](https://www.kaggle.com/c/dog-breed-identification)."
st.markdown(
    """
The data included:
- 10,222 images
- 120 unique breeds
"""
)


df_labels = get_labels()
if st.checkbox("Show raw data"):
    df_labels


if st.checkbox("Show number of images per breed"):
    plot_classes()


"## Training"

"I trained MobileNetV2 on 1000 images first and got an accuracy of 64% on the validation set."

st.markdown(
"""
Take a look at some of the predictions the model made
* Correct breed is at the top of picture, prediction is at the bottom, red denotes it got it wrong.
* Each image is accompanied by graph of top 10 predicted breeds and confidence score.
"""
)

from PIL import Image

image = Image.open("assets/p1.png")

st.image(image, caption="MobileNetV2 predictions")

"Later on, I tranined a new instance of MobileNetV2 on the full dataset (10,222 images)."

"I submitted predictions made on a test dataset to Kaggle and got a [Multi Class Log Loss](https://www.kaggle.com/wiki/MultiClassLogLoss) score of 20.07."

"# Improvements"

"The first model was overfitting. Therefore I switched to EfficientNet-BO in the hope that I could get better results."

"Repeating the same steps as before, I submitted another set of predictions, this time from EfficientNet-BO and got a much better Log Loss score: 0.86 🥳🥳"
