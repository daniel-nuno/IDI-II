#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
plt.style.use("dark_background")
# %%
def Kohonen(data, k, step, max_epoch=500, plotting=False):
    data = data
    k = k
    step = step
    max_epoch = max_epoch
    epoch = 1
    no_changes = False #False means there were changes and while cycle needs to continue 
    #randomly create centroids
    # matrix_cntroids = [ np.random.uniform(low=data[:,i].min(), high=data[:,i].max(), size=(k,1)) for i in range(data.shape[1]) ]
    # matrix_cntroids = np.block(matrix_cntroids)
    # using map
    matrix_cntroids = list(map( lambda l, h: np.random.uniform(low=l, high=h, size=(k,1)), data.min(0), data.max(0) ))
    matrix_cntroids = np.block(matrix_cntroids)
    mtrx_cntrds_copy = matrix_cntroids.copy() #create a copy to compare at every epoch
    # data = np.c_[x,y, np.empty( ( len(x) ) ) ] # matrix data with barrio
    while (no_changes == False) and (epoch < max_epoch):
        matrix_dstnc = []
        vector_barrio = np.array([])

        for row in data:
            vector_dstnc = list(map( lambda row, cntrds: np.sum( (row - cntrds)**2 ), row, matrix_cntroids ))
            vector_dstnc = np.asarray(vector_dstnc, dtype=np.float64)
            barrio = np.argmin(vector_dstnc, axis=0)
            matrix_cntroids[barrio] = matrix_cntroids[barrio] + 1/(step*epoch) * (row - matrix_cntroids[barrio])
            vector_barrio = np.append(vector_barrio, barrio)
            matrix_dstnc.append(vector_dstnc)

        matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64)
        no_changes = np.all( np.round_(mtrx_cntrds_copy) == np.round_(matrix_cntroids) )
        mtrx_cntrds_copy = matrix_cntroids.copy()
        # data[:,2] = vector_barrio
        epoch += 1
        if plotting:
            plt.figure()
            plt.scatter(x, y, c=vector_barrio)
            plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
            for i in range(k):
                plt.annotate("{:.0f}".format(i), (matrix_cntroids[i]), textcoords="offset points", xytext=(0,10))
            plt.show()

    return (np.c_[data, vector_barrio], matrix_cntroids, vector_barrio, epoch)

def EvaluateKohonen(data, matrix_cntroids):
    data = data
    matrix_cntroids = matrix_cntroids
    matrix_dstnc = list(map( lambda cntrds: np.sum( (data - cntrds)**2, axis=1), matrix_cntroids )) #calculates distances
    matrix_dstnc = np.column_stack(matrix_dstnc) # from list to array
    barrio = np.argmin(matrix_dstnc, axis=1) # define barrio
    return barrio

#%% data and values
x = np.linspace(0, 20, 100)
y = (np.sin(x)+2)*2
z = (np.sin(y)+2)*2
plt.scatter(x, y)
k = 4
step = 1
max_epoch=50
data = np.c_[x,y,z]
#%%
data_df, matrix_cntroids, barrio, epochs = Kohonen(data, k, step)
# %%
new_x = np.array([[1],[1]])
new_y = np.array([[0],[5]])
new_data = np.c_[new_x,new_y,z[0:2]]
nuevo_barrio = EvaluateKohonen(new_data, matrix_cntroids)
#%%
plt.figure()
plt.scatter(np.append(x,new_x), np.append(y,new_y), c=np.append(barrio,nuevo_barrio))
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
plt.show()
