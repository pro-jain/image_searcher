import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# Debugging: Print current directory and available files
st.write("Current Directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

# Check if required files exist before loading
if "embeddings.pkl" not in os.listdir() or "filenames.pkl" not in os.listdir():
    st.error("Required files not found! Please check your deployment.")
    st.stop()  # Stop execution if files are missing

# Load embeddings and filenames
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
    filenames = pickle.load(open('filenames.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([model, GlobalMaxPooling2D()])

st.title('Fashion Recommender System')

# Ensure "uploads" folder exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")


def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path  # Return the file path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


def feature_extraction(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        return result / norm(result)  # Normalize features
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None


def recommend(features, feature_list):
    try:
        neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([features])
        return indices
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return None


# Upload and process image
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)

    if file_path:
        # Display uploaded image
        display_image = Image.open(uploaded_file)
        st.image(display_image, caption="Uploaded Image", use_container_width=True)

        # Extract features
        features = feature_extraction(file_path, model)
        if features is not None:
            # Get recommendations
            indices = recommend(features, feature_list)
            if indices is not None:
                # Display recommended images
                st.subheader("Similar Products:")
                cols = st.columns(5)  # 5 columns for 5 images

                for i, col in enumerate(cols):
                    with col:
                        st.image(filenames[indices[0][i]], use_container_width=True)
    else:
        st.error("Some error occurred while uploading the file.")
