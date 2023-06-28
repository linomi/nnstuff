## importing stuff 
from matplotlib import pyplot as plt
from tensorflow.keras.layers import LSTM, Input,Dense
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np

# preprocessing data and labels 
dataset = pd.read_csv("ecg dataset/ecg.csv")
dataset = dataset.to_numpy()
dataset = np.expand_dims(dataset, axis=2)
labels = dataset[:,-1]
dataset = dataset[:,:-1]


#building model 
input_layer = Input((140,1))
lstm = LSTM(50)(input_layer)
output_layer = Dense(1,activation = 'sigmoid')(lstm)
model = Model(input_layer,output_layer)

#fit model to data
model.compile(optimizer = 'adam',loss = 'BCE',metrics='acc')
model.fit(dataset,labels,batch_size = 100,shuffle=True,validation_split = 0.25,epochs = 20)
