#importing stuff
import math
import numpy as np 
from matplotlib import pyplot as plt 
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
import tensorflow as tf
from random import uniform

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
linear_layer = Dense(1,activation='linear',name= 'linear_layer',use_bias= False)(input_layer)
model = Model(input_layer,linear_layer)
model.summary()
model.compile(optimizer ='sgd',loss = 'mse')
history = model.fit(data,target,epochs =100,verbose= True)
#print(history.history["loss"][-10:])

#get weights 
model.get_weights()
