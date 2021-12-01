from cnvrg import Endpoint
from scipy.misc.pilutil import imread, imresize
import numpy as np
from keras.models import load_model

model = load_model('output/imagizer.model.h5')
e = Endpoint()


def predict(file_path):
    x = imread(file_path, mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    x = x.reshape(1, 28, 28, 1)
    x = x.astype('float32')
    x /= 255
    out = model.predict(x)
    e.log_metric("digit", np.argmax(out))
    return str(np.argmax(out))
