import pycaret.classification as pc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os

dir_identifiers = ['blue', 'red', 'green', 'nir', 'gt']
feed_num = 5

dir = os.listdir("/home/jv/Documents/38cloud/38-Cloud_training/train_gt/")
print(len(dir))

channels = {
	'red': pd.DataFrame(columns=['r']),
	'green': pd.DataFrame(columns=['g']),
	'blue': pd.DataFrame(columns=['b']),
	'nir': pd.DataFrame(columns=['nir']),
	'gt': pd.DataFrame(columns=['gt'])
}

for color in dir_identifiers:
	images = os.listdir(f"/home/jv/Documents/38cloud/38-Cloud_training/train_{color}/")
	for i in range(feed_num):
		# TODO: Concatenar as imagens nas linhas de cada canal
		channels[color] = pd.concat([channels[color]], ignore_index=True)

# Juntando as colunas dos dataframes de única coluna (canal)
data = pd.concat([channels[color] for color in channels], axis=1)

setup = pc.setup(data, target='gt')
best_model = pc.compare_models()
pc.save_model(best_model, 'modelo_bom')

holdout_pred = pc.predict_model(best_model)