from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

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
target_image = "_patch_84_5_by_4_LC08_L1TP_003052_20160120_20170405_01_T1.TIF"
image = tiff.imread(f"{color_dir_prefix}red/red{target_image}")

original_height = len(image)
original_width = len(image[0])

for color in colors:
    image = tiff.imread(f"{color_dir_prefix}{color}/{color}{target_image}")
    channels[color] = pd.DataFrame(image.reshape(-1, 1), columns=[color])

# Concatenate the dataframes into a single dataframe
data = pd.concat([channels[color] for color in channels], axis=1)

print("data")
print(data)

# Use the loaded model to make predictions
predicted_df = predict_model(loaded_model, data)

# print(predicted_df['prediction_label'])

for i in range(20, 500):
    print(predicted_df.iloc[i]['prediction_label'])

# Assuming predicted_df is your predicted DataFrame
# Reshape the DataFrame into the original image dimensions
reconstructed_image = predicted_df['prediction_label'].values.reshape(original_height, original_width)

# Convert to grayscale (optional)
# Assuming 0 represents no cloud and 1 represents cloud coverage
reconstructed_image_gray = np.where(reconstructed_image < 0.5, 255, 0)

# Display the reconstructed image
plt.imshow(reconstructed_image_gray)
plt.axis('off')
plt.show()
