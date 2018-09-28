import numpy as np
from keras.models import load_model
# import skimage.io as imgio
import matplotlib.pyplot as plt
from PIL import Image


model = load_model('CNN.h5')

image_path = 'test.png'

'''
img = imgio.imread(image_path, as_gray=True)
imgio.imshow(img)
plt.show()
img = np.array(img)
img = img.reshape((1, 1, 28, 28)).astype('float32')
print(img)
'''

img = Image.open(image_path)
img = img.convert('L')
plt.imshow(img)
plt.show()
img = 255 - np.reshape(img, (1, 1, 28, 28)).astype('float32')
img = img / 255
print(img)

# proba = model.predict(img)
# print(proba)
result = model.predict_classes(img)
print('识别结果：', result[0])
