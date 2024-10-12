import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the trained model
model = load_model('model.h5')

# Load label encoder (label.pkl)
with open('label.pkl', 'rb') as file:
    le = pickle.load(file)

# Streamlit app title and description
st.title("Anime and Cartoon Recognition!")
st.write("This is a simple web app to predict whether the character in the image is an anime or cartoon character.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Function to preprocess the image
def preprocess_image(image):
    """Preprocesses the uploaded image for prediction."""
    img = load_img(image, target_size=(224, 224))  # Resize image to 224x224
    img = img_to_array(img)  # Convert image to array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Function to predict the class of the image
def predict_image(image):
    """Predicts the label for the uploaded image using the trained model."""
    img = preprocess_image(image)
    prediction = model.predict(img)  # Predict directly using the loaded model
    return prediction

# Display uploaded image and make prediction
if uploaded_file is not None:
    # Display the uploaded image in a small size
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    

    # Make prediction
    prediction = predict_image(uploaded_file)
    predicted_class = np.argmax(prediction)  # Get the predicted class index
    predicted_label = le.inverse_transform([predicted_class])  # Decode the prediction

    # Display the prediction
    if predicted_class == 0:
        st.write(f"Prediction: The character in the image is a Cartoon Character.")
    else:
        st.write(f"Prediction: The character in the image is an Anime Character.")
else:
    st.write("Please upload an image to predict.")
