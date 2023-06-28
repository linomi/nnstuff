#importing stuff
import os 
from tensorflow import keras
from tensorflow.keras.layers import Dense,Conv2D,Input
import numpy as np 
from matplotlib import pyplot as plt 
from PIL import Image
from tensorflow.keras.models import Model

# preparing dataset 
dataset_folder = 'yalefaces/'
def face(folder):
    faces = []
    for file_name in os.listdir(folder): 
        face = Image.open(folder+file_name)
        face = np.asarray(face)
        faces.append(face)
    return np.array(faces)
faces = face(dataset_folder)
vetorized_faces = faces.reshape(166,77760)/255
f = vetorized_faces[0].reshape(243,320)*255
f = Image.fromarray(f)
f.show()

#building model 
N = vetorized_faces.shape[0] #Number of data Sample point
F = vetorized_faces.shape[1] #arbitrary input size feature
M = 400 #arbitrary PC feature size 
reconstructed_data = vetorized_faces
data = vetorized_faces
layer = Input(shape =(F,),name='Input_layer')
PC_layer = Dense(1,activation= 'linear',use_bias=False,name='PC_layer')(layer)
output_layer = Dense(F,activation= 'linear',use_bias=False,name='output_layer')(PC_layer)
model = Model(layer,output_layer)
model.summary()
model.compile(optimizer = keras.optimizers.Adam(learning_rate =0.1),loss = "mae")
#extracting PCs
PCs =[]
vectors = []
for i in range(0,M):
    model.fit(data,reconstructed_data,batch_size =int(N/20),epochs=30,verbose =False)
    PC_Model = Model(model.input,model.get_layer('PC_layer').output)
    vectors.append(PC_Model.get_weights())
    #get Pc out of the model an store it in PCs array for visualization purpose
    PCs.append(np.transpose(PC_Model(data)))
    ## to cancel out the effect of ith Pc in reconstructed data for i+1th step ,
    #  we simply subtract each step reconstructed data from original one 
    reconstructed_data = reconstructed_data - np.array(model(data))
PCs = np.array(PCs)
#saving PCs
vectors = np.array(vectors)