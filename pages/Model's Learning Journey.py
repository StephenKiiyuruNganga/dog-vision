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
- 10222 images
- 120 unique breeds
"""
)


df_labels = get_labels()
if st.checkbox("Show raw data"):
    df_labels


if st.checkbox("Show number of samples per breed"):
    plot_classes()


"## Training"

"## Validation"
"## Kaggle Scores"
