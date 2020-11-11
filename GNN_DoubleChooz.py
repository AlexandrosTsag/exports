import keras as k
import spektral as spk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import sparse as sp
from spektral.layers import GraphConv
from spektral.layers.ops import sp_matrix_to_sp_tensor
from sklearn.model_selection import train_test_split

#Parameters
l2_reg = 5e-4
learning_rate = 1e-3
batch_size = 32
epochs = 15
es_patience = 10

#load data

#positions of all 390 inner PMTs in cartesian coordinates
pos = np.load('DCMC_InnerPMTPositions.npy')

#File with scintillation light weighted vertices in cartesian coordinates.
light = np.load('DCMC_PreprocessedVertices.npy')

#true “measured” energy including quenching effects
Y_tr = np.load('DCMC_PreprocessedEQuenchedTrue.npy')

#File containing the true deposited energy, not including quenching effects
y_tr = np.load('DCMC_PreprocessedEdepTrue.npy')

#File containing the actual input data, namely the first hit time and charge for all 390 PMTs for all events
x_tr = np.load('DCMC_PreprocessedDataSet_Charges.npy')

print('pos_shape   :', pos.shape,
    '\nlight_shape :', light.shape,
    '\nY_tr_shape  :', Y_tr.shape,
    '\ny_tr_shape  :', y_tr.shape,
    '\nx_tr_shape  :', x_tr.shape )

#training, validation and test set --> 70%, 15%, 15%
x_tr, x_te, y_tr, y_te = train_test_split(x_tr, y_tr, test_size=0.30, random_state=42)
x_te, x_va, y_te, y_va = train_test_split(x_te, y_te, test_size=0.50, random_state=42)

N = x_tr.shape[-2]      # Number of nodes in the graphs
F = x_tr.shape[-1]      # Node features dimensionality
n_out = 1               # Dimension of the target

print('Dataset splited:', '\nx_tr_shape  :', x_tr.shape,
    '\ny_tr_shape  :', y_tr.shape,
    '\nx_te_shape  :', x_te.shape,
    '\ny_te_shape  :', y_te.shape,
    '\nx_va_shape  :', x_va.shape,
    '\ny_va_shape  :', y_va.shape)

A = np.zeros([390,390])

def distance(x, y, z):
    return np.sqrt(x**2+y**2+z**2)

def minimum(lst, N=16):
    pos_m = []
    for i in range(N):
        minpos = lst.index(min(lst))
        pos_m.append(minpos)
        lst[minpos] = max(lst)
    return pos_m

x = pos.T[0]
y = pos.T[1]
z = pos.T[2]
#print(x)
edges = []
for i in range(390):
    d = []
    for j in range(390):
        dis = distance(x[i]-x[j], y[i]-y[j], z[i]-z[j])
        d.append(dis)
    m = minimum(d, N=11)
    edges.append([(i, k) for k in m])
    for j in m:
        A[i][j] = 1

A = A - np.identity(390)
A = sp.csr_matrix(A, dtype='float32')
fltr = spk.utils.localpooling_filter(A)
fltr = spk.layers.ops.sp_matrix_to_sp_tensor(fltr)

# Model definition

X_in = k.layers.Input(shape = (N,F))
A_in = k.layers.Input(tensor = fltr)

graph_conv = spk.layers.GraphConv(32,
                                 activation = 'elu',
                                 kernel_regularizer = k.regularizers.l2(l2_reg),
                                 use_bias = True)([X_in,A_in])
graph_conv = spk.layers.GraphConv(32,
                                 activation = 'elu',
                                 kernel_regularizer = k.regularizers.l2(l2_reg),
                                 use_bias = True)([graph_conv,A_in])
flatten = k.layers.Flatten()(graph_conv)
fc = k.layers.Dense(512,activation = 'relu')(flatten)
output = k.layers.Dense(n_out)(fc)

# Build Model

model = k.Model(inputs = [X_in, A_in], outputs = output)
optimizer = k.optimizers.Adam(lr = learning_rate)
#
model.compile(optimizer = optimizer,
              loss = 'mse')
model.summary()

#Train model
validation_data = (x_va,y_va)
history = model.fit(x_tr,
          y_tr,
          batch_size = batch_size,
          validation_data = validation_data,
          epochs = epochs,
          callbacks = [k.callbacks.EarlyStopping(patience = es_patience,restore_best_weights = True)])


#Evaluate model
print('Evaluating model.')
eval_results = model.evaluate(x_te,
                              y_te,
                              batch_size = batch_size)


plt.figure(figsize=(10, 10))
epoch = np.linspace(1,epochs,epochs)

plt.plot(epoch, history.history['loss'], label='Train loss')
plt.plot(epoch, history.history['val_loss'], label='Valid loss')
#plt.plot(epoch, te_loss_values, label='Test loss')
plt.grid(True)
plt.legend(prop={'size': 20})
plt.title('Loss of graph neural network for DoubleChooz dataset', fontsize=18)
plt.xticks(np.linspace(1,epochs,epochs))
plt.xlabel('epoch', fontsize=22)
plt.ylabel('Loss', fontsize=22)
plt.savefig('exports/last_Loss1.jpg')

print('Done.\n'
      'Test loss: ', eval_results)
