from pycaret.classification import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os

# Definir diretórios
data_dir = "/home/vitor/Documentos/38cloud/38-Cloud_training/"
colors = ['blue', 'red', 'green', 'nir', 'gt']

# Criar dataframes para cada canal de cor e máscara
channels = {
    'red': pd.DataFrame(),
    'green': pd.DataFrame(),
    'blue': pd.DataFrame(),
    'nir': pd.DataFrame(),
    'gt': pd.DataFrame()
}

feed_num = 1
color_dir_prefix = "/home/jv/Documents/38cloud/38-Cloud_training/train_"

# Iterar sobre os diretórios e arquivos
i = 0
pivot_color = 'green'
good_images = []
offset = 1996

# Extraindo os identificadores das imagens
dir = os.listdir(f"/home/jv/Documents/38cloud/38-Cloud_training/train_{pivot_color}")
for i in range(offset, len(dir)):
	good_images.append(dir[i].replace(pivot_color, "")) # Extraindo os identificadores das imagens
	if i >= feed_num:
		break
	i += 1

print(good_images)

# Iterando igualmente pelos canais das imagens de good_images
for image_name in good_images:
	for color in colors:
		image = tiff.imread(f"{color_dir_prefix}{color}/{color}{image_name}")
		channels[color] = pd.concat([channels[color], pd.DataFrame(image.reshape(-1, 1), columns=[color])], ignore_index=True)

# for color in colors:
# 	i = i+1 
# 	if filename.endswith(".TIF"):
# 		# Ler a imagem TIFF
# 		image = tiff.imread(os.path.join(images_dir, filename))
# 		# Concatenar a imagem no dataframe correspondente
# 		channels[color] = pd.concat([channels[color], pd.DataFrame(image.reshape(-1, 1), columns=[color])], ignore_index=True)
# 	if (i%11) == 10:
# 		print("Lendo imagem ", i)
# 	if i == 500:
# 		break

# Juntar os dataframes de cada canal de cor em um dataframe único
data = pd.concat([channels[color] for color in channels], axis=1)

# Configurar o ambiente
setup(data, target='gt')

# Comparar modelos e selecionar o melhor
best_model = compare_models()

# Salvar o modelo
save_model(best_model, 'modelo_bom')

# Fazer previsões no conjunto de dados de teste
holdout_pred = predict_model(best_model)