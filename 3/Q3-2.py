#importing stuff
import math
import numpy as np 
from matplotlib import pyplot as plt 
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
import tensorflow as tf
from random import uniform
#getting input and output of model 
N = range(0,1000) #number of steps or samples in other word 
#bulding input vector and target 
N = range(0,1000)
xk = [uniform(-10,10) for n in N] # x at step k 
xkk = [uniform(-10,10) for n in N] # x at step k-1
xkkk = [uniform(-10,10) for n in N] #x at step k-2
ykk = [uniform(-10,10) for n in N] # y at step k-1
ykkk = [uniform(-10,10) for n in N] # y at step k-2
yk = [(-6*ykk[n]-8*ykkk[n]+2*xk[n]+4*xkk[n]) for n in N]
data = np.transpose(np.array([xk,xkk,xkkk,ykk,ykkk]))
target = np.transpose(np.array(yk)[np.newaxis])
#building model 
input_layer = Input((5,))
nonlinear_layer = Dense(20,activation='sigmoid',name= 'BOSS_layer')(input_layer)
output_layer  = Dense(1,activation = 'linear',name='linear_layer')(nonlinear_layer)
model = Model(input_layer,output_layer)
model.summary()
model.compile(optimizer ='adam',loss = 'mse')
history = model.fit(data,target,epochs =10000,verbose= False)
print(history.history["loss"][-10:])
h = np.array(history.history['loss']).min()


#inference model
test_data = np.random.rand(10,5)
test_target = np.transpose(np.array([(2*row[0]+4*row[1]-6*row[3]-8*row[4]) for row in test_data])[np.newaxis])
target_hat  = model(test_data)
print(test_target-target_hat)

model.save('narx_model')