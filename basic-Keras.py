#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from operator import add
from functools import reduce
import tensorflow as tf
from tensorflow import keras
import numpy as np


def printAllWeights(model):
    for layer in model.layers:
        print(layer.get_weights())

input2dList = [[1,1,1],[1,1,0],[0,0,0],[0,0,1]]
outputTrainList = [[1],[1],[0],[0]]

test2dList = [[1,1,1],[1,0,1],[0,0,0]]
inputSize = len(input2dList[0])

# define the keras model
model = keras.Sequential()
model.add(keras.layers.Dense(inputSize, input_shape=(inputSize,), activation='relu', kernel_initializer=keras.initializers.Constant(value=0.1)))
model.add(keras.layers.Dense(1, activation='sigmoid', kernel_initializer=keras.initializers.Constant(value=0.5)))

print(model.summary())
print("Weights before training")
printAllWeights(model)

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.SGD(lr=0.1), metrics=['accuracy'])
print("training started")
# needs at least 20 epochs at learning rate 0.1
model.fit([input2dList], [outputTrainList], epochs=20, batch_size=1, verbose=1)

print("Weights after training")
printAllWeights(model)

print("pred started")
predictions = model.predict_proba([test2dList])

print(predictions)





