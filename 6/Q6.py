#importing stuff 
from re import L
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D,UpSampling2D,MaxPooling2D,Input
from tensorflow.keras.models import Model 
import numpy as np 
from matplotlib import pyplot as plt 
from os import listdir
from PIL import Image,ImageOps
import random

#dataset loading and pre processing data 
def data_loader(path):
    dataset_color = []
    dataset_gray = []
    #shuffling pictures
    dir = listdir(path)
    random.shuffle(dir)
    for filename in dir:
        fruit = Image.open(path+filename)
        fruit_color  = np.asarray(fruit)
        dataset_color.append(fruit_color)
        #converting to grayscale 
        fruit_gray = ImageOps.grayscale(fruit)
        fruit_gray = np.asarray(fruit_gray)
        dataset_gray.append(fruit_gray)
    #return the color and  gray datasets and normalizig them 
    return np.array(dataset_color)/255,np.array(dataset_gray)/255
color_dataset,gray_dataset = data_loader(folder)
#reshaping gray dataset 
gray_dataset = gray_dataset.reshape(1000,100,100,1)
print(gray_dataset.shape)

#building model 
input_layer = Input(shape=(100,100,1))
layer = Conv2D(256,(3,3),padding = 'same',activation = 'relu')(input_layer)
layer = MaxPooling2D((2,2))(layer)
layer = Conv2D(128,(3,3),padding = 'same',activation = 'tanh')(layer)
layer = MaxPooling2D((2,2))(layer)
layer = Conv2D(64,(3,3),padding = 'same',activation = 'tanh')(layer)
layer = UpSampling2D((2,2))(layer)
layer = Conv2D(128,(3,3),padding = 'same',activation = 'tanh')(layer)
layer = UpSampling2D((2,2))(layer)
output_layer = Conv2D(3,(3,3),padding = 'same',activation = 'relu')(layer)
model = Model(input_layer,output_layer)
model.compile(optimizer = 'adam',loss = 'mse',metrics = 'acc')
#fiiting model to dataset 
model.fit(train_gray,train_color,batch_size = 10,epochs = 6,validation_split = 0.2)


# model inference 
test = model(test_gray)
test = np.model(test_gray)
for j in len(0,test.shap[0]):
    rec_fruit = (255*(test[j]- test[j].min())/test[j].max()).astype(uint8)
    fruit = (255*test_color[j]).astype(uint8)
    comparison = np.concatenate(rec_fruit,fruit,axis = 0)
    image = Image.fromarray(comparison)
    image.save('test/test{}.jpg'.format(j))

