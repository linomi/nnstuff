#from tensorflow.keras.layers import Input,GlobalAveragePooling2D,Conv2D,Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
from matplotlib import pyplot as plt 
import numpy as np 
from os import listdir
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50,decode_predictions
from scipy.ndimage import zoom


#obtain resnet model
resnet_model = ResNet50()
resnet_model.summary()

# get useful output out of resnet model
convlayer_output = resnet_model.get_layer('conv5_block3_out').output
GAP_layer_output = resnet_model.get_layer('avg_pool').output
prediction_output = resnet_model.get_layer('predictions').output
model = Model(resnet_model.input,[convlayer_output,GAP_layer_output,prediction_output])

data = []
for file in listdir('test'):
  image = Image.open('test/'+file)
  image = image.resize((224,224))
  data.append(np.asarray(image))
data = np.array(data)
j = 12 #input picture
conv_out,GAP_out,predict = model.predict(np.expand_dims(data[j],axis=0))
perceptron_weight,_= resnet_model.get_layer('predictions').weights
perceptron_weight= perceptron_weight[:,np.argmax(predict)]
nine_filter = np.multiply(perceptron_weight,GAP_out[0]) 
nine_filter = np.argpartition(nine_filter,-9,axis = 0)[-9:]
nine_heatmap = conv_out[0,:,:,[nine_filter]]
plt.figure(figsize= (10,10))
for i in range(1,10):
  plt.subplot(3,3,i)
  plt.imshow(data[j])
  plt.imshow(zoom(nine_heatmap[0,i-1],zoom =(224/7)),cmap = 'jet',alpha=0.3)
