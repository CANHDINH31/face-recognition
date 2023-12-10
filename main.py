import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical


classes = ['airplane', 'automoblie', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship','truck']

(Xtrain, ytrain),(Xtest, ytest) = cifar10.load_data()

Xtrain, Xtest = Xtrain/255, Xtest/255
# ytrain, ytest = to_categorical(ytrain), to_categorical(ytest)

# for i in range(50):
#     plt.subplot(5,10,i+1)
#     plt.imshow(Xtrain[i])
#     plt.title(classes[ytrain[i][0]])
#     plt.axis('off')
#
# plt.show()

# model_training_first = models.Sequential([
#     layers.Conv2D(32, (3,3), input_shape = (32,32,3), activation="relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.15),
#
#     layers.Conv2D(64, (3,3), activation="relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Conv2D(128, (3,3), activation="relu"),
#     layers.MaxPool2D((2,2)),
#     layers.Dropout(0.2),
#
#     layers.Flatten(input_shape=(32,32,3)),
#     layers.Dense(1000, activation = 'relu'),
#     layers.Dense(256, activation='relu'),
#     layers.Dense(10, activation='softmax')
# ])
#
# model_training_first.summary()
#
# model_training_first.compile(optimizer ='adam',
#                              loss='categorical_crossentropy',
#                              metrics=['accuracy'])
#
#
# model_training_first.fit(Xtrain, ytrain, epochs=10)
# model_training_first.save('model-cifar10.h5')
#
#
models = models.load_model('model-cifar10.h5')

# pred = models.predict(Xtest[105].reshape(-1,32,32,3))
# print(classes[np.argmax(pred)])
# plt.imshow(Xtest[105])
# plt.show()

# np.random.shuffle((Xtest))

acc = 0
for i in range(100):
    plt.subplot(100, 10, i+1)
    plt.imshow(Xtest[500+i])
    if (np.argmax(models.predict(Xtest[500+i].reshape(-1,32,32,3))) == ytest[500+i][0]):
        acc +=1
    plt.title(classes[np.argmax(models.predict(Xtest[500+i].reshape(-1,32,32,3)))])
    plt.axis("off")

plt.show()
print(acc)
