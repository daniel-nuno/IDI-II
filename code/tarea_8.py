#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use("fivethirtyeight")
plt.style.use("dark_background")
# %%
def KMeans(data, k, max_epoch=500, plotting=False):
    data = data
    k = k
    max_epoch = max_epoch
    epoch = 1
    no_changes = False #False means there were changes and while cycle needs to continue 
    #randomly create centroids
    # matrix_cntroids = [ np.random.uniform(low=data[:,i].min(), high=data[:,i].max(), size=(k,1)) for i in range(data.shape[1]) ]
    # matrix_cntroids = np.block(matrix_cntroids)
    # using map
    matrix_cntroids = list(map( lambda l, h: np.random.uniform(low=l, high=h, size=(k,1)), data.min(0), data.max(0) ))
    matrix_cntroids = np.block(matrix_cntroids)
    # calculate first distances
    matrix_dstnc = list(map( lambda cntrds: np.sum( (data - cntrds)**2, axis=1), matrix_cntroids ))
    #matrix_dstnc = [ np.sum( (data - matrix_cntroids[i])**2, axis=1) for i in range(k) ]
    matrix_dstnc = np.column_stack(matrix_dstnc)
    #calculate first neighbourhood
    barrio = np.argmin(matrix_dstnc, axis=1)
    #append neighbourhood to data matrix
    data = np.c_[data,barrio]
    if plotting:
        plt.figure()
        plt.scatter(x, y, c=barrio)
        plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
        for i in range(k):
            plt.annotate("{:.0f}".format(i), (matrix_cntroids[i]), textcoords="offset points", xytext=(0,10))
        plt.show()
    while (no_changes == False) and (epoch < max_epoch):
        matrix_cntroids = list(map( lambda i: data[data[:,-1] == i, :-1].mean(axis=0), range(k) )) # calculate centroids
        matrix_cntroids = np.array(matrix_cntroids) # from list to array
        matrix_dstnc = list(map( lambda cntrds: np.sum( (data[:,:-1] - cntrds)**2, axis=1), matrix_cntroids )) #calculates distances
        matrix_dstnc = np.column_stack(matrix_dstnc)
        barrio = np.argmin(matrix_dstnc, axis=1) # define barrio
        no_changes = np.all( data[:,-1] == barrio ) # Test whether all array elements along a given axis evaluate to True. if all true no changes made hence break loop
        data[:,-1] = barrio # assign barrio to data frame like array
        epoch += 1
        if plotting:
            plt.figure()
            plt.scatter(x, y, c=barrio)
            plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
            for i in range(k):
                plt.annotate("{:.0f}".format(i), (matrix_cntroids[i]), textcoords="offset points", xytext=(0,10))
            plt.show()

    return (data, matrix_cntroids, barrio, epoch)

def Evaluate_KMeans(x, y, matrix_cntroids):
    x = x
    y = y
    matrix_cntroids = matrix_cntroids
    matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ] # calculate distances
    matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64) # from list to array
    barrio = np.argmin(matrix_dstnc, axis=0) # define barrio
    return barrio

#%%
x = np.linspace(0, 20, 100)
y = (np.sin(x)+2)*2
z = (np.sin(y)+2)*2
plt.scatter(x, y)
k = 5
step = 1
max_epoch=50
data = np.c_[x,y]
#%%
data_df, matrix_cntroids, barrio, epochs = KMeans(data, k, plotting=False)
# matplotlib
plt.figure()
plt.scatter(x, y, c=barrio)
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
for i in range(k):
    plt.annotate("{:.0f}".format(i), (matrix_cntroids[i]), textcoords="offset points", xytext=(0,10))
plt.show()
#%% seaborn
data_df = pd.DataFrame(data_df, columns=["x", "y", "barrio"])
matrix_cntroids_df = pd.DataFrame(matrix_cntroids, columns=["x", "y"])
sns.scatterplot(data=data_df, x="x", y="y", hue="barrio")
sns.scatterplot(x=matrix_cntroids[:,0], y=matrix_cntroids[:,1], color="red")
#%% k means evaluation
new_x = np.array([[1]])
new_y = np.array([[0]])
nuevo_barrio = Evaluate_KMeans(new_x, new_y, matrix_cntroids)
plt.figure()
plt.scatter(np.append(x,new_x), np.append(y,new_y), c=np.append(barrio,nuevo_barrio))
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
plt.show()


#matrix_cntroids = [ np.random.uniform(low=data[:,i].min(), high=data[:,i].max(), size=(k,1)) for i in range(data.shape[1]) ]
#matrix_cntroids = np.asarray(matrix_cntroids, dtype=np.float64) # from list to array