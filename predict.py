from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

# Load the trained model
loaded_model = load_model('modelo_bom_2img')

# Define directories
colors = ['blue', 'red', 'green', 'nir']
color_dir_prefix = "/home/jv/Documents/38cloud/38-Cloud_test/test_"
#color_dir_prefix = "C:\\Documentos\\38cloud\\38-Cloud_test\\test_"

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
original_image = tiff.imread(f"{color_dir_prefix}red/red{target_image}")

original_height = len(original_image)
original_width = len(original_image[0])

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
reconstructed_image_gray = np.where(reconstructed_image >= 0.5, 255, 0)

# Create a blank canvas with the same dimensions as the original image
height, width = reconstructed_image_gray.shape[:2]
bgr_reconstructed = np.zeros((height, width, 3), dtype=np.uint8)  # Create a black canvas

# Convert the grayscale image to a 3-channel image (assuming it's a grayscale image)
bgr_reconstructed[:, :, 0] = reconstructed_image_gray  # Copy the grayscale image to all channels
bgr_reconstructed[:, :, 1] = reconstructed_image_gray
bgr_reconstructed[:, :, 2] = reconstructed_image_gray

# Salvando a máscara gerada
cv2.imwrite('cloud_mask.TIF', bgr_reconstructed)

# Read the cloud mask image (binary image with clouds as white regions)
cloud_mask = cv2.imread('cloud_mask.TIF', cv2.IMREAD_GRAYSCALE)

# Find contours in the cloud mask image
maskContours, _ = cv2.findContours(cloud_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around the contours
print("DESENHANDO OS CONTORNOS")
for contour in maskContours:
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)
    print(f"BB = ({x}, {y}, {w}, {h})")
    
    # Draw the bounding box on the original image (assuming you have the original image)
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)

# Display the image with bounding boxes
print("SALVANDO A IMAGEM COM OS BOUNDING BOXES PREVISTOS")
cv2.imwrite('bounded_image.TIF', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou