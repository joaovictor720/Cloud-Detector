from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff

# Load the trained model
loaded_model = load_model('modelo_bom')

# Define directories
colors = ['blue', 'red', 'green', 'nir']
color_dir_prefix = "/home/jv/Documents/38cloud/38-Cloud_test/test_"

# Create dataframes for each color channel and mask
channels = {
    'red': pd.DataFrame(),
    'green': pd.DataFrame(),
    'blue': pd.DataFrame(),
    'nir': pd.DataFrame(),
    'gt': pd.DataFrame()
}

# Load the new image and preprocess it
target_image = "_patch_169_9_by_9_LC08_L1TP_029044_20160720_20170222_01_T1.TIF"
for color in colors:
    image = tiff.imread(f"{color_dir_prefix}{color}/{color}{target_image}")
    channels[color] = pd.DataFrame(image.reshape(-1, 1), columns=[color])

# Concatenate the dataframes into a single dataframe
data = pd.concat([channels[color] for color in channels], axis=1)

print("data")
print(data)

# Use the loaded model to make predictions
predictions = predict_model(loaded_model, data)

print(predictions['prediction_label'])
