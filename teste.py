# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

# Load TIF image
image_path = "last_red.TIF"
red = tiff.imread(image_path)

image_path = "last_green.TIF"
green = tiff.imread(image_path)

image_path = "last_blue.TIF"
blue = tiff.imread(image_path)

image_path = "last_nir.TIF"
nir = tiff.imread(image_path)

print('red')
print(red[0])

# Convert flattened array to DataFrame
df_r = pd.DataFrame(red.flatten(), columns=['R'])
df_g = pd.DataFrame(green.flatten(), columns=['G'])
df_b = pd.DataFrame(blue.flatten(), columns=['B'])
df_nir = pd.DataFrame(nir.flatten(), columns=['NIR'])

df = pd.concat([df_r, df_g, df_b, df_nir], axis=1)

# Display DataFrame
print(df.head())