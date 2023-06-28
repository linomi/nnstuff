from re import L
import tensorflow as tf 
from tensorflow.keras.layers import Conv2D,UpSampling2D,MaxPooling2D,Input
from tensorflow.keras.models import Model 
import numpy as np 
from matplotlib import pyplot as plt 
from os import listdir
from PIL import Image,ImageOps
import random
from matplotlib import pyplot as plt

#dataset loading and pre processing data 
def data_loader(dataset_path):
    dataset_gray = []
    #shuffling pictures
    dir = listdir(dataset_path)
    for fruit_type in dir:
        file = listdir(dataset_path+'/'+fruit_type)
        #random.shuffle(file)
        for filename in file:
            #print(dataset_path+'/'+fruit_type+'/'+filename)
            fruit = Image.open(dataset_path+'/'+fruit_type+'/'+filename)
            fruit = ImageOps.grayscale(fruit)
            fruit_gray  = np.asarray(fruit)
            dataset_gray.append(fruit_gray)
    #return the dataset 
    return np.array(dataset_gray)/255
data = np.expand_dims(data_loader('dataset'),3)
#building model 
input_layer = Input((100,100,1))
layer = Conv2D(50,(6,6),activation = 'tanh',padding ='same')(input_layer)
layer = MaxPooling2D(pool_size = (4,4),strides =(4,4))(layer)
layer = Conv2D(50,(5,5),activation = 'tanh',padding= 'same')(layer)
layer = MaxPooling2D(pool_size = (4,4),strides =(6,6))(layer)
layer = Conv2D(1,(3,3),activation = 'tanh')(layer)
encoder_layer = Conv2D(1,(1,2),activation = 'elu',name = 'encoder_end')(layer)
layer = UpSampling2D((5,5))(encoder_layer)
layer = Conv2D(50,(3,3),activation = 'tanh',padding = 'same')(layer)
layer = UpSampling2D((5,5))(layer)
layer = Conv2D(100,(5,5),activation = 'tanh',padding = 'same')(layer)
layer = UpSampling2D((2,4))(layer)
layer = Conv2D(1,(6,6),activation = 'sigmoid',padding = 'same')(layer)
model = Model(input_layer,layer)
#model.summary()
model.compile(optimizer = 'sgd',loss = 'mae')
model.fit(data,data,batch_size = 400,epochs = 50,verbose = False)

#showing latent layer output 
encoder =Model(model.input,model.get_layer("encoder_end").output)
encoded = encoder.predict(data)
encoded = encoded[:,:,0,0]
x = encoded[:,0]
y = encoded[:,1]
plt.scatter(x,y)
plt.show()

#kmean clustering 
from sklearn.cluster import KMeans
performance_index = []
for i in range(1,5):
  kmean = KMeans(i,init = "random",n_init=10,random_state = 0)
  kmean.fit(encoded)
  performance_index.append(kmean.inertia_)
#elbow graph to choose correct cluster number
plt.plot(range(1,5),performance_index)
plt.xlabel = 'number of cluster'
plt.ylabel = 'performance'
plt.show()

#test result with blueberry picture
kmean = KMeans(2,init = "random",n_init=10,random_state = 0)
label = kmean.fit_predict(encoded)
#test function 
path = 'test_blue_berry.jpg'
image = Image.open(path)
image =np.array([np.asarray(image)]) 
encoded_test = np.array([encoder.predict(image)[0,:,0,0]])
cluster = kmean.predict(encoded_test)
cluster = cluster - 1 
print("it belong to cluster{}".format(cluster))
