from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2

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

def cv_bb_to_iou_bb(bb):
	return [bb[0], bb[1], bb[0]+bb[3], bb[1]+bb[2]]

def load_image(img_id, base_dir="/home/jv/Documents/38cloud/38-Cloud_training/train_", extension="TIF"):
	colors = ['blue', 'red', 'green', 'nir']
	#color_dir_prefix = "C:\\Documentos\\38cloud\\38-Cloud_test\\test_"
	original_image = tiff.imread(f"{base_dir}red/red{img_id}.{extension}")
	channels = {
		'red': pd.DataFrame(),
		'green': pd.DataFrame(),
		'blue': pd.DataFrame(),
		'nir': pd.DataFrame(),
		'gt': pd.DataFrame()
	}
	original_height = len(original_image)
	original_width = len(original_image[0])

	for color in colors:
		image = tiff.imread(f"{base_dir}{color}/{color}{img_id}.{extension}")
		channels[color] = pd.DataFrame(image.reshape(-1, 1), columns=[color])

	# Concatenate the dataframes into a single dataframe
	data = pd.concat([channels[color] for color in channels], axis=1)
	return data, original_height, original_width

# Retorna a imagem original com os retângulos desenhados, e as coordenadas dos retângulos
# da máscara e do ground-truth
def handle_bbs(mask_id, gt_id, original_id, base_dir="/home/jv/Documents/38cloud/38-Cloud_training", extension="TIF"):
	mask_path = f"{base_dir}/masks/mask{mask_id}.{extension}"
	gt_path = f"{base_dir}/train_gt/gt{gt_id}.{extension}"
	original_path = f"{base_dir}/train_red/red{original_id}.{extension}"

	# Lendo as máscaras
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
	original_image = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)

	# Find contours in the cloud mask image
	mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	gt_contours, _ = cv2.findContours(gt_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Desenhando os bounding boxes da máscara gerada
	mask_boxes = []
	for contour in mask_contours:
		x, y, w, h = cv2.boundingRect(contour)
		mask_boxes.append(cv_bb_to_iou_bb(cv2.boundingRect(contour)))
		cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 0, 255), 1)
	
	# Desenhando os bounding boxes do gabarito
	gt_boxes = []
	for contour in gt_contours:
		x, y, w, h = cv2.boundingRect(contour)
		gt_boxes.append(cv_bb_to_iou_bb(cv2.boundingRect(contour)))
		cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	return original_image, mask_boxes, gt_boxes

def get_ious(mask_boxes, gt_boxes):
	ious = []
	for i in range(min(len(mask_boxes), len(gt_boxes))):
		print(f"Calculando ({mask_boxes[i]}) / ({gt_boxes[i]})")
		ious.append(bb_intersection_over_union(mask_boxes[i], gt_boxes[i]))
	return ious

# Load the trained model
original_image_id = "_patch_267_13_by_15_LC08_L1TP_029040_20160720_20170222_01_T1"
loaded_model = load_model('modelo_bom_2img')

# Concatenate the dataframes into a single dataframe
loaded_image, original_height, original_width = load_image(original_image_id)

print("data")
print(loaded_image)

# Use the loaded model to make predictions
predicted_df = predict_model(loaded_model, loaded_image)

# Assuming predicted_df is your predicted DataFrame
# Reshape the DataFrame into the original image dimensions
predicted_matrix = predicted_df['prediction_label'].values.reshape(original_height, original_width)

# Convert to grayscale (optional)
# Assuming 0 represents no cloud and 1 represents cloud coverage
binarized_matrix = np.where(predicted_matrix >= 0.5, 255, 0)

# Create a blank canvas with the same dimensions as the original image
height, width = binarized_matrix.shape[:2]
matrix_3channel = np.zeros((height, width, 3), dtype=np.uint8)  # Create a black canvas

# Convert the grayscale image to a 3-channel image (assuming it's a grayscale image)
matrix_3channel[:, :, 0] = binarized_matrix  # Copy the grayscale image to all channels
matrix_3channel[:, :, 1] = binarized_matrix
matrix_3channel[:, :, 2] = binarized_matrix

# Salvando a máscara gerada
base_dir = "/home/jv/Documents/38cloud/38-Cloud_training"
cv2.imwrite(f"{base_dir}/masks/mask{original_image_id}.TIF", matrix_3channel)

bounded_image, mask_bbs, gt_bbs = handle_bbs(gt_id=original_image_id, mask_id=original_image_id, original_id=original_image_id)

print("IoUs:")
print(get_ious(mask_bbs, gt_bbs))

# Display the image with bounding boxes
print("SALVANDO A IMAGEM COM OS BOUNDING BOXES PREVISTOS")
cv2.imwrite('bounded_image.TIF', bounded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
