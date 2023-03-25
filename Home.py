import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        filepath="models/20230214-0648-full-data-EfficientnetB0-Adam.h5",
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    return model


"# Dog Vision 🐶"

model = load_model()
# img = tf.keras.utils.plot_model(model, "assets/EfficientNetB0.png")
# model.summary()
# config = model.get_config()
# config
