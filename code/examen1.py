#%%
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen2México.jpg")
col, row = image_input.size
col = col - 1
row = row - 1
a = np.asarray(image_input)
#%%
step_row = row/100
step_col = col/100
pixeles = []
for i in range(0, 100, 4):
    pixeles.append(a[round(0+step_row*i),round(0+step_col*i)])
    pixeles.append(a[round(row-step_row*i), round(col-step_col*i)])
    pixeles.append(a[round(row-step_row*i),round(0+step_col*i)])
    pixeles.append(a[round(0+step_row*i),round(col-step_col*i)])

# %%
pixeles = []
for i in range(0, round(row/2), 2):
    for j in range(0, col, 2):
        pixeles.append(a[i,j])
        pixeles.append(a[-i,-j])
# %%
pixeles = []
for i in range(0, round(row/2), 2):
    for j in range(0, col, 2):
        pixeles.append(a[i,j])
        pixeles.append(a[round(row/2)+i,-j])
# %%
