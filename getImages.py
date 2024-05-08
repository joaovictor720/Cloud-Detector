import os
import shutil
import sys

def encontrar_arquivo(nome_arquivo, pasta):
    for nome in os.listdir(pasta):
        if nome_arquivo in nome:
            return os.path.join(pasta, nome)
    return None

def copiar_arquivos(nome_arquivo, origens, destino):
    for origem in origens:
        caminho_arquivo = encontrar_arquivo(nome_arquivo, origem)
        if caminho_arquivo:
            shutil.copy(caminho_arquivo, destino)

def main(nome_arquivo):
    pasta_origem = "C:\\Documentos\\38cloud\\38-Cloud_training"
    pastas = ["train_blue", "train_red", "train_green", "train_nir", "train_gt"]
    pasta_destino = "used_images/38-cloud_training/"

    # Cria a pasta de destino se não existir
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    for pasta in pastas:
        caminho_pasta = os.path.join(pasta_origem, pasta)
        copiar_arquivos(nome_arquivo, [caminho_pasta], pasta_destino+pasta)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Por favor, passe o nome do arquivo como parâmetro.")
        sys.exit(1)
    nome_arquivo = sys.argv[1]
    main(nome_arquivo)