#importing stuff
import math
import numpy as np 
from matplotlib import pyplot as plt 
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model

#getting input and output of model 
N = range(0,8) #number of steps or samples in other word 
x = [math.sin(n*np.pi/8) for n in N]
y = [0,0]
for n in N[2:]: 
    nth_output = -6*y[n-1] - 8*y[n-2]+2*x[n]+4*x[n-1]
    y.append(nth_output)
#getting y[k],y[k-1],y[k-2],x[k],x[k-1],x[k-2]
x_k = x[2:] # x at step k 
x_kk = x[1:-1] # x at step k-1
x_kkk = x[:-2]   #x at step k-2 
y_k = y[2:] #y at step k 
y_kk = y[1:-1] #y ate step k-1
y_kkk = y[:-2]   #y at step k-2 
#building dataset
data = np.array([x_k,x_kk,x_kkk,y_kk,y_kkk]).reshape(len(N)-2,5)
target = np.array(y_k)[np.newaxis].reshape(len(N)-2,1)

#building model 
input_layer = Input((5,))
nonlinear_layer = Dense(20,activation='sigmoid',name= 'BOSS_layer')(input_layer)
output_layer  = Dense(1,activation = 'linear',name='linear_layer')(nonlinear_layer)
model = Model(input_layer,output_layer)
model.summary()

model.compile(optimizer ='rmsprop',loss = 'mse')
history = model.fit(data,target,epochs =10,verbose= True)

