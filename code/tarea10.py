#%%
from PIL import Image as im
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def KMeans(data, k, max_epoch=500, plotting=False):
    data = data
    k = k
    max_epoch = max_epoch
    epoch = 1
    no_changes = False #False means there were changes and while cycle needs to continue 
    matrix_cntroids = list(map( lambda l, h: np.random.uniform(low=l, high=h, size=(k,1)), data.min(0), data.max(0) )) #randomly create centroids
    matrix_cntroids = np.block(matrix_cntroids) #to array
    matrix_dstnc = list(map( lambda cntrds: np.sum( (data - cntrds)**2, axis=1), matrix_cntroids )) # calculate first distances
    matrix_dstnc = np.column_stack(matrix_dstnc) #to array
    barrio = np.argmin(matrix_dstnc, axis=1) #calculate first neighbourhood
    data = np.c_[data,barrio] #append neighbourhood to data matrix

    while (no_changes == False) and (epoch < max_epoch):
        matrix_cntroids = list(map( lambda i: data[data[:,-1] == i, :-1].mean(axis=0) if np.any(np.unique(barrio) == i) else matrix_cntroids[i], 
                                    range(k) )) # calculate centroids if there is nothing assigned then not calculate new centroid and keep old
        matrix_cntroids = np.array(matrix_cntroids) # from list to array
        matrix_dstnc = list(map( lambda cntrds: np.sum( (data[:,:-1] - cntrds)**2, axis=1), matrix_cntroids )) #calculate distance
        matrix_dstnc = np.column_stack(matrix_dstnc) # from list to array
        barrio = np.argmin(matrix_dstnc, axis=1) # define barrio
        no_changes = np.all( data[:,-1] == barrio ) # Test whether all array elements along a given axis evaluate to True. if all true no changes made hence break loop
        data[:,-1] = barrio # assign barrio to data frame like array
        epoch += 1

    return (data, matrix_cntroids, barrio, epoch)

def Kohonen(data, k, step, max_epoch=500, plotting=False):
    data = data
    k = k
    step = step
    max_epoch = max_epoch
    epoch = 1
    no_changes = False #False means there were changes and while cycle needs to continue 
    matrix_cntroids = list(map( lambda l, h: np.random.uniform(low=l, high=h, size=(k,1)), data.min(0), data.max(0) )) #randomly create centroids
    matrix_cntroids = np.block(matrix_cntroids) #to array
    mtrx_cntrds_copy = matrix_cntroids.copy()  #create a copy to compare at every epoch
    while (no_changes == False) and (epoch < max_epoch):
        matrix_dstnc = []
        vector_barrio = np.array([])

        for row in data:
            #vector_dstnc = list(map( lambda row, cntrds: np.sum( (row - cntrds)**2, axis=1 ), row, matrix_cntroids )) #calculate distance
            #vector_dstnc = np.asarray(vector_dstnc, dtype=np.float64) #to array
            vector_dstnc = np.sum((row - matrix_cntroids)**2,1)
            barrio = np.argmin(vector_dstnc, axis=0) # define barrio
            matrix_cntroids[barrio] = matrix_cntroids[barrio] + 1/(step*epoch) * (row - matrix_cntroids[barrio]) #reposition centroids
            vector_barrio = np.append(vector_barrio, barrio)
            matrix_dstnc.append(vector_dstnc)

        matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64)
        no_changes = np.all( np.round_(mtrx_cntrds_copy) == np.round_(matrix_cntroids) )
        mtrx_cntrds_copy = matrix_cntroids.copy()
        epoch += 1

    return (np.c_[data, vector_barrio], matrix_cntroids, vector_barrio, epoch)

def EvaluateKohonen(data, matrix_cntroids):
    data = data
    matrix_cntroids = matrix_cntroids
    matrix_dstnc = list(map( lambda cntrds: np.sum( (data - cntrds)**2, axis=1), matrix_cntroids )) #calculates distances
    matrix_dstnc = np.column_stack(matrix_dstnc) # from list to array
    barrio = np.argmin(matrix_dstnc, axis=1) # define barrio
    return barrio

#%% ----------------------------------------------------------------------
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen1Deutschland-1.jpg")
x,y = image_input.size
a = np.asarray(image_input)
#%% kmeans
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    data_df, matrix_cntroids, barrio, epochs = KMeans(data, k)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen1Deutschland-1_kmeans_k") + str(i) + str(".jpg")
    image_output.save(img_txt)
# %%
#%% kohones
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    step = 1
    data_df, matrix_cntroids, barrio, epochs = Kohonen(data, k, step)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen1Deutschland-1_kohones_k") + str(i) + str(".jpg")
    image_output.save(img_txt)



#%% ----------------------------------------------------------------------
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen2México.jpg")
x,y = image_input.size
a = np.asarray(image_input)
#%% kmeans
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    data_df, matrix_cntroids, barrio, epochs = KMeans(data, k)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen2México_kmeans_k") + str(i) + str(".jpg")
    image_output.save(img_txt)
# %%
#%% kohones
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    step = 1
    data_df, matrix_cntroids, barrio, epochs = Kohonen(data, k, step)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen2México_kohones_k") + str(i) + str(".jpg")
    image_output.save(img_txt)
# %%


#%% ----------------------------------------------------------------------
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen3Gandhi.jpg")
x,y = image_input.size
a = np.asarray(image_input)
#%% kmeans
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    data_df, matrix_cntroids, barrio, epochs = KMeans(data, k)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen3Gandhi_kmeans_k") + str(i) + str(".jpg")
    image_output.save(img_txt)
# %%
#%% kohones
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    step = 1
    data_df, matrix_cntroids, barrio, epochs = Kohonen(data, k, step)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen3Gandhi_kohones_k") + str(i) + str(".jpg")
    image_output.save(img_txt)

#%% ----------------------------------------------------------------------
image_input = im.open("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen.jpg")
x,y = image_input.size
a = np.asarray(image_input)
#%% kmeans
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    data_df, matrix_cntroids, barrio, epochs = KMeans(data, k)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen_kmeans_k") + str(i) + str(".jpg")
    image_output.save(img_txt)
# %%
#%% kohones
for i in [2,3,10]:
    data = a.reshape((x*y, 3))
    k = i
    step = 1
    data_df, matrix_cntroids, barrio, epochs = Kohonen(data, k, step)
    data_output = list(map( lambda barrio: matrix_cntroids[barrio], barrio.astype(int) ))
    data_output = np.array(data_output, np.uint8)
    data_output = data_output.reshape((y,x,3))
    image_output = im.fromarray(data_output) #este último array tiene que ser de enteros, uint8
    img_txt = str("C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/"
                    "idi_ii/imagen4Lauterbrunnen_kohones_k") + str(i) + str(".jpg")
    image_output.save(img_txt)