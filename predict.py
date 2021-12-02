from cnvrg import Endpoint
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import urllib

model = load_model('/input/train/output/imagizer.model.h5')
e = Endpoint()


def predict(file_url):
    path = urllib.request.urlopen(file_url).read()
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    image_tensor = np.vstack([x])
    classes = model.predict(image_tensor)
    print(classes)
    print(classes[0])
    if classes[0] > 0.5:
        return "human"
    else:
        return "horse"
