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
    target = []
    for file_name in os.listdir(folder): 
        if file_name.split('.')[0] == 'subject01':
            face = Image.open(folder+file_name)
            face = np.asarray(face)
            faces.append(face)
            target.append(0)
        elif file_name.split('.')[0] == 'subject02':
            face = Image.open(folder+file_name)
            face = np.asarray(face)
            faces.append(face)
            target.append(1)
    return np.array(faces),np.array(target)
faces,target = face(dataset_folder)

#loading vectors 
vectors = np.load('face_PCs.npy')
 
#reshaping PCA vectors and Dataset Vectors 
vectors = vectors.reshape(77760,400)
faces = faces.reshape(23,77760)/255

#reducing dataset features with Principal component analysis 
Faces_reduces_dimension = np.matmul(faces,vectors)

#building model 
in_layer = Input(shape=(400))
layer = Dense(200,activation = 'relu')(in_layer)
layer = Dense(50,activation = 'relu')(layer)
output_layer = Dense(1,activation = 'sigmoid')(layer)
model = Model(in_layer,output_layer)
model.compile(optimizer=keras.optimizers.Adam(learning_rate = 0.1e-3),loss = 'bce')
keras.utils.plot_model(model,show_shapes = True,show_layer_names = True)

#fitting model to the dataset
model.fit(Faces_reduces_dimension,target,epochs = 100,validation_split = 0.2,shuffle = True)

#testing model classifier
out = ['subject0','subject1']
x = out[int(np.heaviside(model(np.array([Faces_reduces_dimension[0]]))-0.5,1))]
print(x)




