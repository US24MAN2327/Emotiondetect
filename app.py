import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import plotly.graph_objects as go
from keras.models import load_model
import requests
import os
import time

# Set page config
st.set_page_config(page_title="Emotion Classifier", layout="wide")

# Custom CSS for animations and styling
st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0;}
    to {opacity: 1;}
}
.fade-in {
    animation: fadeIn 1.5s;
}
.stApp {
    background: linear-gradient(to right, #4e54c8, #8f94fb);
}
</style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown('<h1 class="fade-in" style="text-align: center; color: white;">Emotion Classifier</h1>', unsafe_allow_html=True)

# Function to download and load the model
@st.cache_resource  # Cache the model so it's not re-downloaded multiple times
def load_model_from_dropbox(url):
    model_path = "model.h5"
    if not os.path.exists(model_path):
        response = requests.get(url, stream=True)
        with open(model_path, 'wb') as f:
            f.write(response.content)
    return load_model(model_path)

# Dropbox public link to the model (ensure the link ends with dl=1 for direct download)
dropbox_url = 'https://www.dropbox.com/scl/fi/0vjfk1isg73feb9wg80vn/seq.keras?rlkey=0xekvnj2fsa9qlunon5yca31m&st=e4xaxs2y&dl=0'

with st.spinner('Downloading and loading the model...'):
    try:
        resnet34 = load_model_from_dropbox(dropbox_url)
        st.success('Model loaded successfully!')
    except Exception as e:
        st.error(f'Error loading model: {e}')

# Function to predict emotion
def predict_emotion(img):
    classname = ["angry", "sad", "happy"]
    
    # Ensure the image has 3 channels
    if img.shape[-1] == 4:
        img = img[:,:,:3]  # Remove alpha channel if present
    elif len(img.shape) == 2:
        img = tf.stack([img, img, img], axis=-1)  # Convert grayscale to RGB
    
    img = tf.image.resize(img, (224, 224))  # Resize the image to 224x224 for ResNet input
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    
    # Perform prediction
    predicted = resnet34.predict(img)
    
    # Get the predicted emotion
    emotion = classname[np.argmax(predicted[0])]
    probabilities = predicted[0]
    
    return emotion, probabilities

# File uploader for the image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict Emotion'):
        with st.spinner('Analyzing...'):
            time.sleep(2)
            img_array = np.array(image)
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
            img_array = tf.image.convert_image_dtype(img_array, dtype=tf.float32)

            emotion, probabilities = predict_emotion(img_array)

        # Display result with animation
        st.markdown(f'<h2 class="fade-in" style="text-align: center; color: white;">Predicted Emotion: {emotion.capitalize()}</h2>', unsafe_allow_html=True)

        # Create a donut chart for probabilities
        fig = go.Figure(data=[go.Pie(labels=['Angry', 'Happy', 'Sad'],
                                     values=probabilities,
                                     hole=.3,
                                     marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1'])])
        fig.update_layout(title_text="Emotion Probabilities",
                          font=dict(size=16, color="white"),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown('<p class="fade-in" style="text-align: center; color: white;">Created with ❤️ by AI Hub</p>', unsafe_allow_html=True)
