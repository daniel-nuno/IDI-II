#%%
import sympy as sp
import numpy as np
import pandas as pd

#%%
x = np.array(range(1,101))
sigma = np.random.rand(100)
y = x + 3*sigma
n = 100
# %%
# puede ser de menos infinito a infinito tambien. pero voy a calcular el promedio de y y limitar un rango pequeño.
# si b optima se acerca al maximo o minimo entonces ajusto el resultado
a = np.linspace(-3,3,1000)
# cada x crece 3*[0, 1), o sea entre 0 y 3.
# en promedio 3*0.5=1.5 por que es una distribución uniforme.
# lo más probable es se encuentre entre de 0 a 4.
b = np.linspace(-3,3,1000)

#%%
#ahora a encontrar todas las rectas con su respectivo error para graficar el la superficia
# forma 1 con aburridos for
matrix_y_estimada = []
matrix_errores = []
E = np.zeros((len(a), len(b)))
#%%
for i in range(len(a)):
    for j in range(len(b)):
        matrix_y_estimada.append(( a[j] + b[i]*x ))
        matrix_errores.append( (matrix_y_estimada[-1] - y)**2 )
        E[i, j] = (matrix_errores[-1]).mean()
#%%
A, B = np.meshgrid(a, b)
E_pd = pd.DataFrame(E, index=a, columns=b)

# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#%%

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(A, B, E, cmap=cm.viridis)

# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf)

plt.show() #show 3d surface

plt.contourf(A, B, E, 15, cmap=cm.viridis) #show contour

#%%
#gradiente

a_t = 0
b_t = 0
alpha = 0.0001
exactitud = 10**-3
iteraciones = 0
error = 1
while error >= exactitud:
    iteraciones += 1
    y_pred = b_t*x + a_t
    df_db = (-2/n)*sum(x*(y - y_pred))
    df_da = (-2/n)*sum(y - y_pred)
    error = np.sqrt(df_da**2 + df_db**2)
    error_cuadratico = ((y_pred - y )**2).mean() / 2
    b_t = b_t - alpha * df_db
    a_t = a_t - alpha * df_da
    
print(b_t, a_t, error, error_cuadratico)

#%%
#grafica
y_pred = b_t*x + a_t
plt.style.use('ggplot')
plt.scatter(x, y)
plt.plot([min(x), max(x)], [min(y_pred), max(y_pred)], color='blue')
plt.show()
# %%
data = pd.read_excel("C:/Users/nuno/Downloads/tareaRGD.xlsx", "data")
# %%
#X = np.array(data[['x1', 'x2', 'x3']])
#Y = np.array(data['y'])
X = data[['x1', 'x2', 'x3']]
Y = np.array(data['y'])
beta = np.zeros(4) #np.array([0,0,0,0])
alpha = 0.00001
exactitud = 10**-3
iteraciones = 0
error = 1
n = len(X)
while error >= exactitud and iteraciones < 10000:
    iteraciones += 1
    y_pred = (np.array(beta[0]) +
                np.array(beta[1]*np.array(X.iloc[:,[0]]).reshape(64)) +
                np.array(beta[2]*np.array(X.iloc[:,[1]]).reshape(64)) +
                np.array(beta[3]*np.array(X.iloc[:,[2]]).reshape(64)))
    #y_pred = beta[0] + ( beta[1:4] * X ).sum(1)
    dB_0 = (-2/n)*sum(y_pred - Y)
    dB_1 = (-2/n)*sum(np.array(X.iloc[:,[0]]).reshape(64) * (y_pred - Y ))
    dB_2 = (-2/n)*sum(np.array(X.iloc[:,[1]]).reshape(64) * (y_pred - Y ))
    dB_3 = (-2/n)*sum(np.array(X.iloc[:,[2]]).reshape(64) * (y_pred - Y ))
    error = np.sqrt(dB_0**2 + dB_1**2 + dB_2**2 + dB_3**2)
    error_cuadratico = ((y_pred - Y)**2).mean() / 2

    beta[0] = beta[0] - alpha*dB_0
    beta[1] = beta[1] - alpha*dB_1
    beta[2] = beta[2] - alpha*dB_2
    beta[3] = beta[3] - alpha*dB_3
    
print(beta, error, error_cuadratico)
#X[:,::3]

##%
X = np.array(data[['x1', 'x2', 'x3']])
Y = np.array(data['y'])


#%%
# Building the model
X = x
Y = y
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 10000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    Y_pred = m*X + c  # The current predicted value of Y
    D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
    D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    
print (m, c)
# %%

exactitud = 10**-3
iteraciones = 0
an=0
bn=0
Bn = np.array([an, bn])
lista_A = [an]
lista_B = [bn]
alpha = 0.00001

df_da = ( (an + bn*x) - y ).mean()
df_db = (( (an + bn*x) - y ) * x).mean()
gradiente_E = np.array([df_da, df_db])
error = np.sqrt(df_da**2 + df_db**2)
error_cuadratico = (((an + bn*x) - y )**2).mean() / 2

while error >= exactitud and iteraciones < 5:
    iteraciones += 1
    Bn_1 = Bn - alpha * gradiente_E
    lista_A.append( Bn_1[0] )
    lista_B.append( Bn_1[1] )
    df_da = ( (Bn_1[0] + Bn_1[1]*x) - y ).mean()
    df_db = (( (Bn_1[0] + Bn_1[1]*x) - y ) * x).mean()
    gradiente_E = np.array([df_da, df_db])
    error = np.sqrt(df_da**2 + df_db**2)
    error_cuadratico = (((Bn_1[0] + Bn_1[1]*x) - y )**2).mean() / 2
    Bn = Bn_1
    print(Bn_1)
    print(error, error_cuadratico)