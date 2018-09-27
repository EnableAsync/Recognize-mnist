import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import  Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPool2D, Flatten
from keras.optimizers import Adam

np.random.seed(1337)  # for reproducibility

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# data pre-processing
X_train = X_train.reshape(-1, 1, 28, 28) / 255  # 正常化 归一化  1 表示图片高度
X_test = X_test.reshape(-1, 1, 28, 28) / 255  # 正常化 归一化
Y_train = np_utils.to_categorical(Y_train, num_classes=10)
Y_test = np_utils.to_categorical(Y_test, num_classes=10)
print(X_train.shape)

# build a CNN
model = Sequential()

# Convolution layer 1
model.add(Convolution2D(
    filters=32,
    kernel_size=(5, 5),
    padding='same',  # padding method
    input_shape=(1, 28, 28),  # channels height width
))
model.add(Activation('relu'))

# Pooling layer1 (max pooling)
model.add(MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='same',  # padding method
))

# Convolution layer 2
model.add(Convolution2D(64, (5, 5), padding='same'))
model.add(Activation('relu'))

# Pooling layer2 (max pooling)
model.add(MaxPool2D(pool_size=(2, 2), padding='same'))

# Fully connected layer 1
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2
model.add(Dense(10))
model.add(Activation('softmax'))

# optimizer
adam = Adam(lr=1e-4)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print('训练中……')
model.fit(X_train, Y_train, epochs=5, batch_size=32)

print('测试中……')
loss, accuracy = model.evaluate(X_test, Y_test)

print('\n测试误差：', loss)
print('\n测试准确率：', accuracy)

model.save('CNN.h5')
