from pycaret.classification import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import os

# Definir diretórios
data_dir = "/home/vitor/Documentos/38cloud/38-Cloud_training/"
dir_identifiers = ['blue', 'red', 'green', 'nir', 'gt']

# Criar dataframes para cada canal de cor e máscara
channels = {
    'red': pd.DataFrame(),
    'green': pd.DataFrame(),
    'blue': pd.DataFrame(),
    'nir': pd.DataFrame(),
    'gt': pd.DataFrame()
}

# Iterar sobre os diretórios e arquivos
for color in dir_identifiers:
	images_dir = os.path.join(data_dir, f"train_{color}")
	print("Lendo imagens cor " + color )
	i = 0
	for filename in os.listdir(images_dir):
		i = i+1 
		if filename.endswith(".TIF"):
			# Ler a imagem TIFF
			image = tiff.imread(os.path.join(images_dir, filename))
			# Concatenar a imagem no dataframe correspondente
			channels[color] = pd.concat([channels[color], pd.DataFrame(image.reshape(-1, 1), columns=[color])], ignore_index=True)
		if (i%11) == 10:
			print("Lendo imagem ", i)
		if i == 500:
			break

             

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