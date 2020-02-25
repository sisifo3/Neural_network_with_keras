import tensorflow as tf
from tensorflow import keras
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
import numpy

#fix random seed for repoducibility
seed = 7
numpy.random.seed(seed)

#==========================================
from google.colab import files
dataset = files.upload()

#=======================================
file = tf.keras.utils
dataset = pd.read_csv('animo_507.csv')

print(dataset)
#=============================

X = dataset.iloc[:,0:2]
Y = dataset.iloc[:,2:3]
print(X)
print(Y)

#===============================

#create a model
model = Sequential()
model.add(Dense(12, input_dim=2, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

#====================================================

#compile de mode
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#====================================================
#fit the model
scores = model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

