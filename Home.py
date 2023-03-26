import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


@st.cache_data
def get_labels():
    return pd.read_csv("train/labels.csv")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        filepath="models/20230214-0648-full-data-EfficientnetB0-Adam.h5",
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    return model


@st.cache_data
def process_img(_img):
    image = tf.image.decode_jpeg(_img, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[224, 224])
    return image


@st.cache_data
def create_batches(_img):
    data_set = tf.data.Dataset.from_tensor_slices(_img)
    data_set = data_set.map(process_img)
    batches = data_set.batch(32)
    return batches


@st.cache_data
def get_name(prediction):
    return unique_breeds[np.argmax(prediction)]


"# Dog Vision 🐶"

model = load_model()
df_labels = get_labels()
labels = np.array(df_labels["breed"])
unique_breeds = np.unique(labels)

file = st.file_uploader(
    label="Upload an image of a dog",
    type=["png", "jpg"],
)

if file is not None:
    user_img = file.read()
    step_1 = st.text(f"Processing '{file.name}'... 🔁")
    img = None
    img = process_img(user_img)

    if img is not None:
        step_1.text(f"Processing '{file.name}'... ✅")
        # st.write(img.shape)
        step_2 = st.text(f"Batching '{file.name}'... 📦")
        batches = None
        data_set = tf.data.Dataset.from_tensor_slices(img)
        data_set = data_set.map(process_img)
        batches = data_set.batch(32)

        if batches is not None:
            step_2.text(f"Batching '{file.name}'... ✅")
            # st.write(batches.element_spec)
            step_3 = st.text(f"Analyzing canine features of '{file.name}'... 🤔")
            pred = None
            pred = model.predict(batches, verbose=1)

            if pred is not None:
                step_3.text(f"Analyzing canine features of '{file.name}'... ✅")
                label = get_name(pred)

                if label:
                    st.image(img.numpy(), caption=label)
