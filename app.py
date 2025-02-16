import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle
from PIL import Image

# Load ResNet50 Model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Function to extract features
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert grayscale images to RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# Get valid image filenames
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
filenames = []

for file in os.listdir('images'):
    file_path = os.path.join('images', file)
    if file.lower().endswith(valid_extensions):
        try:
            with Image.open(file_path) as img:
                img.load()  # Fully load the image to check for issues
            filenames.append(file_path)
        except (IOError, OSError):
            print(f"Skipping invalid image file: {file_path}")

# Extract features
feature_list = []

for file in tqdm(filenames, desc="Extracting Features"):
    feature_list.append(extract_features(file, model))

# Save embeddings and filenames
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))

print(f"✅ Process completed! Extracted {len(feature_list)} feature vectors.")
