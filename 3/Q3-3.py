from tensorflow.keras.models import load_model
import numpy as np

model = load_model('narx_model')
w = model.get_weights()
print(w[0][0])

