import random

# Definir diretórios
data_dir = "/content/drive/MyDrive/used_images/38-Cloud_training/"
colors = ['blue', 'red', 'green', 'nir', 'gt']

# Definir a semente (seed) para replicabilidade
random.seed(22)

# Criar dataframes para cada canal de cor e máscara
channels = {
    'red': pd.DataFrame(),
    'green': pd.DataFrame(),
    'blue': pd.DataFrame(),
    'nir': pd.DataFrame(),
    'gt': pd.DataFrame()
}

feed_num = 13
color_dir_prefix = "/content/drive/MyDrive/used_images/38-Cloud_training/train_"

# Inicializando variáveis
i = 0
pivot_color = 'green'
good_images = []
offset = 0

# Extraindo os identificadores das imagens
dir = os.listdir(f"/content/drive/MyDrive/used_images/38-Cloud_training/train_{pivot_color}")
for i in range(offset, len(dir)):
    good_images.append(dir[i].replace(pivot_color, "")) # Removendo o nome da cor
    if i >= feed_num:
        break
    i += 1

print(good_images)
i = 0
# Iterando igualmente pelos canais das imagens de good_images
for image_name in good_images:
    i += 1
    print("upando imagem", i)
    gt_image = tiff.imread(f"{color_dir_prefix}gt/gt{image_name}")  # Carregar a máscara gt
    num_cloud_pixels = np.sum(gt_image == 255)  # Contar o número de pixels de nuvem
    num_non_cloud_pixels = np.sum(gt_image == 0)  # Contar o número de pixels de não-nuvem
    num_samples = min(num_cloud_pixels, num_non_cloud_pixels)  # Número mínimo de amostras para balancear
    cloud_pixels_indices = np.argwhere(gt_image == 255)  # Índices dos pixels de nuvem
    non_cloud_pixels_indices = np.argwhere(gt_image == 0)  # Índices dos pixels de não-nuvem
    sampled_cloud_indices = random.sample(list(cloud_pixels_indices), num_samples)  # Selecionar amostras aleatórias de pixels de nuvem
    sampled_non_cloud_indices = random.sample(list(non_cloud_pixels_indices), num_samples)  # Selecionar amostras aleatórias de pixels de não-nuvem
    print("número de pixel com nuvens:", num_cloud_pixels)
    print("número de pixel sem nuvens:", num_non_cloud_pixels)
    print("número de pixel a pegar:", num_samples)
    for color in colors:
        image = tiff.imread(f"{color_dir_prefix}{color}/{color}{image_name}")  # Carregar os dados do canal de cor
        cloud_pixels_data = [image[i[0], i[1]] for i in sampled_cloud_indices]  # Dados dos pixels de nuvem
        non_cloud_pixels_data = [image[i[0], i[1]] for i in sampled_non_cloud_indices]  # Dados dos pixels de não-nuvem
        cloud_df = pd.DataFrame(cloud_pixels_data, columns=[color])  # DataFrame dos pixels de nuvem
        non_cloud_df = pd.DataFrame(non_cloud_pixels_data, columns=[color])  # DataFrame dos pixels de não-nuvem
        channels[color] = pd.concat([channels[color], cloud_df, non_cloud_df], ignore_index=True)  # Concatenar os DataFrames de pixels de nuvem e não-nuvem

# Juntar os dataframes de cada canal de cor em um dataframe único
data = pd.concat([channels[color] for color in channels], axis=1)

# Configurar o ambiente
setup(data, target='gt')

# Comparar modelos e selecionar o melhor
best_model = compare_models()

# Salvar o modelo
save_model(best_model, 'modelo_bom')
