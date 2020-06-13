from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.optimizers import SGD
import keras
from keras.layers.advanced_activations import LeakyReLU
import sys

def main():
    mndata = MNIST(sys.argv[1])
    xtrain, ytrain = mndata.load_training()
    xtest, ytest = mndata.load_testing()
    xtrain = np.array(xtrain)
    ytrain = keras.utils.to_categorical(ytrain, 10)
    xtest = np.array(xtest)
    original = np.array(ytest)
    ytest = keras.utils.to_categorical(ytest, 10)
    xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1)
    xtest = xtest.reshape(xtest.shape[0], 28, 28, 1)
    xtrain = xtrain.astype('float')
    xtest = xtest.astype('float')
    xtrain = xtrain/255
    xtest = xtest/255

    model2 = Sequential()
    model2.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu',input_shape=(28,28,1)))
    model2.add(MaxPooling2D(pool_size=(2, 2)))
    model2.add(Dropout(0.5))
    model2.add(Flatten())
    model2.add(Dense(units=128, activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(units=10, activation='softmax'))
    optimizer = SGD(lr=0.003, momentum=0.9)
    model2.compile(loss=keras.losses.categorical_crossentropy,optimizer=optimizer, metrics=['accuracy'])
    model2.fit(x=xtrain,y=ytrain,batch_size=64,epochs=20)

    temp2 = model2.predict(xtest)
    pred2 = np.argmax(np.round(temp2),axis=1)

    print("Predicted digit ")
    for i in pred2:
        print(i)

if __name__ == "__main__":
	main()