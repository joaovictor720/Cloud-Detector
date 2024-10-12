from pycaret.classification import *
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import scipy
from sys import argv

def bb_intersection_over_union(boxA, boxB):	
	# Determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interW = xB - xA + 1
	interH = yB - yA + 1

	# Eliminando bbs que não se sobrepõem
	if interW <= 0 or interH <= 0 :
		return -1.0

	interArea = interW * interH
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	iou = interArea / float(boxAArea + boxBArea - interArea)
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
def handle_bbs_from_image(mask_id, gt_id, original_id, base_dir="/home/jv/Documents/38cloud/38-Cloud_training", extension="TIF"):
	mask_path = f"./masks/{model_name}/mask{mask_id}.{extension}"
	gt_path = f"{base_dir}/train_gt/gt{gt_id}.{extension}"
	original_path = f"{base_dir}/train_red/red{original_id}.{extension}"

	# Lendo as máscaras
	mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
	gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
	original_image = cv2.imread(original_path, cv2.IMREAD_COLOR)

	# Obtendo os contornos da máscara e gabarito
	mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	gt_contours, _ = cv2.findContours(gt_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Desenhando os bounding boxes da máscara gerada
	mask_boxes = []
	mask_boxes_cv = []
	for contour in mask_contours:
		bb = cv2.boundingRect(contour)
		mask_boxes_cv.append(bb)
		mask_boxes.append(cv_bb_to_iou_bb(bb))
		cv2.rectangle(original_image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 255, 0), 1)
	
	# Desenhando os bounding boxes do gabarito
	gt_boxes = []
	gt_boxes_cv = []
	for contour in gt_contours:
		bb = cv2.boundingRect(contour)
		gt_boxes_cv.append(bb)
		gt_boxes.append(cv_bb_to_iou_bb(bb))
		cv2.rectangle(original_image, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]), (0, 200, 200), 1) # Desenhando um bb de pred

	# Obtendo os matches e IoUs entre os bounding boxes
	matched_gt_idxs, matched_mask_idxs, ious, pred_truths = match_bbs(np.array(mask_boxes), np.array(gt_boxes), IOU_THRESH=0.5)

	# Desenhando os pares previsão-gt na imagem
	for i in range(len(matched_gt_idxs)):
		bb = mask_boxes_cv[matched_gt_idxs[i]]
		bb = gt_boxes_cv[matched_mask_idxs[i]]

	return original_image, ious, pred_truths

def match_bbs(gt_boxes, pred_boxes, IOU_THRESH=0.5):
	'''
	Given sets of true and predicted bounding-boxes,
	determine the best possible match.
	Parameters
	----------
	bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
	The number of bboxes, N1 and N2, need not be the same.

	Returns
	-------
	(idxs_true, idxs_pred, ious, labels)
	idxs_true, idxs_pred : indices into gt and pred for matches
	ious : corresponding IOU value of each match
	labels: vector of 0/1 values for the list of detections
	'''
	n_true = gt_boxes.shape[0]
	n_pred = pred_boxes.shape[0]
	MAX_DIST = 1.0
	MIN_IOU = 0.0

	# NUM_GT x NUM_PRED
	iou_matrix = np.zeros((n_true, n_pred))
	for i in range(n_true):
		for j in range(n_pred):
			iou_matrix[i, j] = bb_intersection_over_union(gt_boxes[i,:], pred_boxes[j,:])

	if n_pred > n_true:
		# there are more predictions than ground-truth - add dummy rows
		diff = n_pred - n_true
		iou_matrix = np.concatenate( (iou_matrix, np.full((diff, n_pred), MIN_IOU)), axis=0)

	if n_true > n_pred:
		# more ground-truth than predictions - add dummy columns
		diff = n_true - n_pred
		iou_matrix = np.concatenate( (iou_matrix, np.full((n_true, diff), MIN_IOU)),axis=1)

	# call the Hungarian matching
	idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

	if (not idxs_true.size) or (not idxs_pred.size):
		ious = np.array([])
	else:
		ious = iou_matrix[idxs_true, idxs_pred]

	# remove dummy assignments
	sel_pred = idxs_pred<n_pred
	idx_pred_actual = idxs_pred[sel_pred] 
	idx_gt_actual = idxs_true[sel_pred]
	ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
	sel_valid = (ious_actual > IOU_THRESH)
	label = sel_valid.astype(int)

	return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label 

if __name__ == "__main__":
	# Parâmetros
	model_name = "model_14_Balanced"
	original_image_id = "_patch_236_12_by_16_LC08_L1TP_002053_20160520_20170324_01_T1"

	for i in range(len(argv)):
		if argv[i] == "-m" and i+1 < len(argv):
			model_name = argv[i+1]
		elif argv[i] == "-i" and i+1 < len(argv):
			original_image_id = argv[i+1]

	# Carregando o modelo treinado
	print(f"Loading model: {model_name}")
	loaded_model = load_model(f'./models/{model_name}')

	# Carregando a imagem de teste em um dataframe a partir de seus canais separados
	print(f"Loading image from ID: {original_image_id}")
	loaded_image, original_height, original_width = load_image(original_image_id)

	# Fazendo predições
	print(f"Making predictions...")
	predicted_df = predict_model(loaded_model, loaded_image)

	# Reorganizando o dataframe às dimensões da imagem original
	print(f"Parsing predictions...")
	predicted_matrix = predicted_df['prediction_label'].values.reshape(original_height, original_width)

	# Binarizando as predições
	binarized_matrix = np.where(predicted_matrix >= 0.5, 255, 0)

	# Criando um canvas vazio (preto) com as mesmas dimensões da imagem original
	height, width = binarized_matrix.shape[:2]
	matrix_3channel = np.zeros((height, width, 3), dtype=np.uint8)

	# Convertendo a matriz binária (máscara) em uma imagem real (de 3 canais iguais)
	matrix_3channel[:, :, 0] = binarized_matrix
	matrix_3channel[:, :, 1] = binarized_matrix
	matrix_3channel[:, :, 2] = binarized_matrix

	# Salvando a máscara real gerada
	print(f"Saving predicted mask in: /masks/{model_name}/mask{original_image_id}.TIF")
	tiff.imwrite(f"./masks/{model_name}/mask{original_image_id}.TIF", matrix_3channel)

	# TODO: Exibindo métricas de testes