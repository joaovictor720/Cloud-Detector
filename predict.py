from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff

# Load the trained model
loaded_model = load_model('modelo_bom')

# Define directories
colors = ['blue', 'red', 'green', 'nir', 'gt']
color_dir_prefix = "/home/jv/Documents/38cloud/38-Cloud_training/train_"

# Create dataframes for each color channel and mask
channels = {
    'red': pd.DataFrame(),
    'green': pd.DataFrame(),
    'blue': pd.DataFrame(),
    'nir': pd.DataFrame(),
    'gt': pd.DataFrame()
}

# Load the new image and preprocess it
image_name = "your_image_name_without_extension"
for color in colors:
    image = tiff.imread(f"{color_dir_prefix}{color}/{color}{image_name}")
    channels[color] = pd.DataFrame(image.reshape(-1, 1), columns=[color])

# Concatenate the dataframes into a single dataframe
data = pd.concat([channels[color] for color in channels], axis=1)

# Use the loaded model to make predictions
predictions = predict_model(loaded_model, data)

# Extract the predicted cloud coverage from the predictions
cloud_coverage = predictions['Label']

# Display or use the predicted cloud coverage
print(cloud_coverage)
