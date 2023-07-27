#%%
import pandas as pd
import numpy as np
#%%
def train_one_layer_perceptron(training_data, training_output, L, a, alpha, accuracy, max_epochs):
    # INITIALIZE values and functions for one hidden layer
    a = a #a = inclinación de la sigmoide tiene que ser mayor a 0
    alpha = alpha #alpha learning rate tiene que ser entre 0 y 1
    L = L #number of neurons per layer
    accuracy = accuracy
    error = 1
    
    x = training_data #Q * N matrix. Q = rows, N = columns
    d = training_output #Q * M matrix. Q = rows, M = columns
    N = training_data.shape[1] #number of Xs = entries columns
    M = training_output.shape[1] #number of Ys = number of columns outputs
    Q = training_data.shape[0] #different inputs with known outputs = number of rows

    epoch_count = 0
    np.random.seed(1)
    weights_ItoH = np.random.uniform(-1, 1, (L, N)) #L x N size
    weights_HtoO = np.random.uniform(-1, 1, (M, L)) #M x L size

    while error > accuracy and (epoch_count < max_epochs):
        for row in range(Q):
            #feedforward
            net_h = weights_ItoH @ np.reshape(x[row,:], (N,1))
            y_h = logistic(net_h,a)
            net_o = weights_HtoO @ np.reshape(y_h, (L,1))
            y_o = logistic(net_o,a)

            #backforward
            delta_o = np.reshape( d[row,:] - np.reshape(y_o, (1,M)), (M,1)) * (y_o * (1 - y_o))
            delta_h = (y_o * (1 - y_o)) * (np.transpose(weights_HtoO) @ delta_o)
            weights_HtoO += alpha * (delta_o * np.transpose(y_h))
            weights_ItoH += alpha * (delta_h * training_data[row,:])
            
            #error
            error = np.linalg.norm(delta_o)
        epoch_count +=1
    return (weights_HtoO, weights_ItoH, error, epoch_count)

def logistic(x,a):
    return 1/(1 + np.exp(-a*x))

def logistic_deriv(x,a):
    return a*logistic(x,a) * (1 - logistic(x,a))

def evaluate_one_layer_perceptron(input_matrix, weights_HtoO, weights_ItoH, a):
    # INITIALIZE values
    x = input_matrix
    weights_HtoO = weights_HtoO
    weights_ItoH = weights_ItoH
    a = a
    N = x.shape[1] #number of Xs = entries columns
    Q = x.shape[0] #different inputs with known outputs = number of rows

    output_matrix = []
    output_row = []
    y_matrix = []

    #cicle through rows to evaluate according to weights
    for row in range(Q):
        net_h = weights_ItoH @ np.reshape(x[row,:], (N,1))
        y_h = logistic(net_h,a)
        net_o = weights_HtoO @ np.reshape(y_h, (L,1))
        y_o = logistic(net_o,a)
        # from y's hidden to outputs define if is 0 or 1 and append to rows outputs
        for y in y_o:
            if y > 0.5:
                output_row.append(1)
            else:
                output_row.append(0)

        output_matrix.append(output_row) #append row outputs in a matrix
        output_row = [] #clear to be used again
        y_matrix.append(y_o) #append return

    output_matrix = np.asarray(output_matrix, dtype=np.float64)
    y_matrix = np.asarray(y_matrix, dtype=np.float64)

    return np.block([x, output_matrix]), output_matrix, y_matrix

#%%
# DEFINE training and test
data = pd.read_excel('C:/Users/nuno/OneDrive - ITESO/Ciencia de Datos/IDI II/PercMultAplicado_tarea7.xlsx')

data['carga_al_salario'] = data.Mensualidad / data['Ingreso mensual']
data.Monto = data.Monto / 300000
data['Antigüedad laboral (meses)'] = data['Antigüedad laboral (meses)'] / 180
data['Mora'] = data['Mora'].replace('SI', 1)
data['Mora'] = data['Mora'].replace('NO', 0)
data = data.drop(['Entidad', 'Plazo (años)', 'Tasa anual', 'Mensualidad', 'Ingreso mensual'], axis = 1)

training_data = data.sample(frac=0.7)
test_data = data.drop(training_data.index)

training_output = training_data[['Mora']]
training_output = np.asarray(training_output, dtype=np.float64)
training_data = training_data[['Monto', 'Antigüedad laboral (meses)', 'carga_al_salario']]
training_data = np.asarray(training_data,dtype=np.float64)

test_output = test_data[['Mora']]
test_output = np.asarray(test_output, dtype=np.float64)
test_data = test_data[['Monto', 'Antigüedad laboral (meses)', 'carga_al_salario']]
test_data = np.asarray(test_data, dtype=np.float64)

#%%
base_limit = np.sum(training_output == 0) / len(training_output)
base_limit
#%%
a = 10
alpha = 0.1
L = 20
accuracy = 0.0001
max_epochs = 100000

weights_HtoO, weights_ItoH, error, epoch_count = train_one_layer_perceptron(training_data,
                                                                            training_output,
                                                                            L,
                                                                            a,
                                                                            alpha,
                                                                            accuracy,
                                                                            max_epochs)

x_and_y, transformed_y, direct_y = evaluate_one_layer_perceptron(test_data,
                                                            weights_HtoO,
                                                            weights_ItoH,
                                                            a)
# %%
vector_accuracy = test_output == transformed_y
accuracy_validation = vector_accuracy.sum() / len(test_data)
accuracy_validation

less_sure = np.sum(direct_y[direct_y > 0.1] < 0.9) / len(test_data)
less_sure