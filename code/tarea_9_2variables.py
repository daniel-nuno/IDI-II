#%%
#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.twodim_base import triu_indices_from
import seaborn as sns
import pandas as pd
plt.style.use("fivethirtyeight")
plt.style.use("dark_background")
# %%
def Kohonen(x, y, k, step, max_epoch=500, plotting=False):
    x = x
    y = y
    k = k
    step = step
    max_epoch = max_epoch
    data = np.c_[x,y, np.empty( ( len(x) ) ) ]
    epoch = 1
    no_changes = False #False means there were changes and while cycle needs to continue 
    #randomly create centroids
    matrix_cntroids = np.block([ np.random.uniform(low=x.min(), high=x.max(), size=(k,1)),
                                np.random.uniform(low=y.min(), high=y.max(), size=(k,1)) ])
    # calculate first distances
    matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ]
    matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64)
    while (no_changes == False) and (epoch < max_epoch):
        matrix_dstnc = []
        vector_barrio = np.array([])

        for row in data:
            vector_dstnc = [ (row[0] - matrix_cntroids[i,0])**2 + (row[1] - matrix_cntroids[i,1])**2 for i in range(k) ]
            vector_dstnc = np.asarray(vector_dstnc, dtype=np.float64)
            barrio = np.argmin(vector_dstnc, axis=0)
            matrix_cntroids[barrio,0] = matrix_cntroids[barrio,0] + 1/(step*epoch) * (row[0] - matrix_cntroids[barrio,0])
            matrix_cntroids[barrio,1] = matrix_cntroids[barrio,1] + 1/(step*epoch) * (row[1] - matrix_cntroids[barrio,1])
            vector_barrio = np.append(vector_barrio, barrio)
            matrix_dstnc.append(vector_dstnc)

        matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64)
        no_changes = np.all( data[:,2] == vector_barrio )
        data[:,2] = vector_barrio
        epoch += 1
        if plotting:
            plt.figure()
            plt.scatter(x, y, c=vector_barrio)
            plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
            for i in range(k):
                plt.annotate("{:.0f}".format(i), (matrix_cntroids[i]), textcoords="offset points", xytext=(0,10))
            plt.show()

    return (data, matrix_cntroids, vector_barrio, epoch)

def EvaluateKohonen(x, y, matrix_cntroids):
    x = x
    y = y
    matrix_cntroids = matrix_cntroids
    matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ] # calculate distances
    matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64) # from list to array
    barrio = np.argmin(matrix_dstnc, axis=0) # define barrio
    return barrio

#%% data and values
x = np.linspace(0, 20, 100)
y = (np.sin(x)+2)*2
plt.scatter(x, y)
k = 2
step = 1

data_df, matrix_cntroids, barrio, epochs = Kohonen(x, y, k, step, plotting=True)
# %%
new_x = np.array([[1]])
new_y = np.array([[0]])
nuevo_barrio = EvaluateKohonen(new_x, new_y, matrix_cntroids)
plt.figure()
plt.scatter(np.append(x,new_x), np.append(y,new_y), c=np.append(barrio,nuevo_barrio))
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
plt.show()
# %%
