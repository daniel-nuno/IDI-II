#%%
import numpy as np
import plotnine
import matplotlib.pyplot as plt
# %%
def KMeans(x, y, k, max_iter=500, plotting=False):
    x = x
    y = y
    k = k
    max_iter = max_iter
    data = np.c_[x,y]
    iter = 1
    #no changes = False means there were changes and while cycle needs to continue 
    no_changes = False
    #randomly create centroids
    matrix_cntroids = np.block([ np.random.uniform(low=x.min(), high=x.max(), size=(k,1)),
                                np.random.uniform(low=y.min(), high=y.max(), size=(k,1)) ])
    # calculate first distances
    matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ]
    matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64)
    #calculate first neighbourhood
    barrio = np.argmin(matrix_dstnc, axis=0)
    #append neighbourhood to data frame like array
    data = np.c_[data,barrio]
    while (no_changes == False) and (iter < max_iter):
        matrix_cntroids = [ data[data[:,2] == i, :-1].mean(axis=0) for i in range(k) ] # calculate centroids
        matrix_cntroids = np.asarray(matrix_cntroids, dtype=np.float64) # from list to array
        matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ] # calculate distances
        matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64) # from list to array
        barrio = np.argmin(matrix_dstnc, axis=0) # define barrio
        no_changes = np.all( data[:,2] == barrio ) # Test whether all array elements along a given axis evaluate to True. if all true no changes made hence break loop
        data[:,2] = barrio # assign barrio to data frame like array
        iter += 1
        if plotting:
            plt.figure()
            plt.scatter(x, y, c=barrio)
            plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
            plt.show()

    return (data, matrix_cntroids, barrio, iter)

def Evaluate_KMeans(x, y, matrix_cntroids):
    x = x
    y = y
    matrix_cntroids = matrix_cntroids
    matrix_dstnc = [ (x - matrix_cntroids[i,0])**2 + (y - matrix_cntroids[i,1])**2 for i in range(k) ] # calculate distances
    matrix_dstnc = np.asarray(matrix_dstnc, dtype=np.float64) # from list to array
    barrio = np.argmin(matrix_dstnc, axis=0) # define barrio
    return barrio

#%%
x = np.linspace(0, 50, 100)
y = np.sin(x)*10
plt.style.use("fivethirtyeight")
plt.scatter(x, y)
k = 4
max_iter = 500
#%%
data, matrix_cntroids, barrio, iter = KMeans(x, y, k)
plt.figure()
plt.scatter(x, y, c=barrio)
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
plt.show()
#%%
new_x = np.array([[1]])
new_y = np.array([[0]])
nuevo_barrio = Evaluate_KMeans(new_x, new_y, matrix_cntroids)

plt.figure()
plt.scatter(np.append(x,new_x), np.append(y,new_y), c=np.append(barrio,nuevo_barrio))
plt.scatter(matrix_cntroids[:,0], matrix_cntroids[:,1], c='r')
plt.show()
# %%
