import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical


classes = ['airplane', 'automoblie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

(Xtrain, ytrain),(Xtest, ytest) = cifar10.load_data()

Xtrain, Xtest = Xtrain/255, Xtest/255
ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

# for i in range(50):
#     plt.subplot(5,10,i+1)
#     plt.imshow(Xtrain[i])
#     plt.title(classes[ytrain[i][0]])
#     plt.axis('off')
#
# plt.show()

model_training_first = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000, activation = 'relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model_training_first.summary()

model_training_first.compile(optimizer ='SGD',
                             loss='categorical_crossentropy',
                             metrics=['accuracy'])


model_training_first.fit(Xtrain, ytrain, epochs=10)
model_training_first.save('model-cifar10.h5')
