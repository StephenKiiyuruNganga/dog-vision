import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from PIL import Image

# page configurations
st.set_page_config(
    page_title="Dog Vision",
    page_icon="🐶",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        filepath="models/20230214-0648-full-data-EfficientnetB0-Adam.h5",
        custom_objects={"KerasLayer": hub.KerasLayer},
    )
    return model


@st.cache_data
def get_labels():
    return pd.read_csv("train/labels.csv")


"# Dog Vision 🐶"

"This app uses a machine learning model to detect the breed of a dog in a given image. Go ahead and try it out! 😀"

model = load_model()
df_labels = get_labels()
labels = np.array(df_labels["breed"])
unique_breeds = np.unique(labels)

uploaded_file = st.file_uploader(
    label="Upload an image of a dog",
    type=["png", "jpg"],
)

if uploaded_file is not None:
    # Read the image using PIL
    img = Image.open(uploaded_file)
    step_1 = st.text(f"Processing '{uploaded_file.name}'... 🔁")

    # Resize the image to the required size
    img = img.resize((224, 224))

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Normalize the image pixel values to be in the range of [0, 1]
    img_array = img_array / 255.0

    # Add a batch dimension to the array
    img_array = np.expand_dims(img_array, axis=0)
    step_1.text(f"Processing '{uploaded_file.name}'... ✅")

    # Use the loaded model to make predictions on the preprocessed image
    step_2 = st.text(f"Analyzing '{uploaded_file.name}'... 🤔")
    prediction = model.predict(img_array, verbose=1)
    label = unique_breeds[np.argmax(prediction)]
    step_2.text(f"Analyzing '{uploaded_file.name}'... ✅\n")

    # Display the converted image and label
    # left_col, right_col = st.columns(2)
    # top_10_idx = prediction.argsort()[::-1][:10]
    # print(top_10_idx)
    # top_10_preds = prediction[top_10_idx]
    # top_10_labels = unique_breeds[top_10_idx]

    # with left_col:
    #     st.image(img_array, caption=label)

    # with right_col:
    #     top_10_labels

    st.image(img_array)
    confidence = np.max(prediction) * 100
    icon = "😀" if confidence > 60 else "🤔"
    f"I am **{confidence:.0f}%** confident that the dog's breed is **{label.capitalize()}** {icon}"

    "# Future Updates"

    st.markdown(
        """
        Introduce another machine learning model on top of this one which will detect whether a given image is actually a dog or not.
        """
    )
