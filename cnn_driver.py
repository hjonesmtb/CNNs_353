import string
import random
from random import randint
import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend


WIDTH = 400
HEIGHT = 100
CROP = 100

#Encodes an array of states
def one_hot(Y):
	Y_encoded = []

	for state in Y:

		state = int(state)
		encoding = np.zeros(10,)    # array to hold one_hot encoding for 1 frame.
		#print(state)
		encoding[state] = 1

		Y_encoded.append(encoding)

	return np.asarray(Y_encoded)

# get frames and their labeled corrections
def split(f,cmd):
	img = cv2.imread("/home/fizzer/Pictures/twist_cmds2/" + f)
	img = cv2.resize(img,(WIDTH,HEIGHT))

	lower_white = np.array([0,0,230])
	upper_white = np.array([255,255,255])

	img = cv2.inRange(img,lower_white,upper_white)


	return img, float(cmd)

def get_images(folder = '/home/fizzer/Pictures/twist_cmds2/'):
	frames = os.listdir(folder)

	X = []
	Y = []

	for file in frames:
		if "_test" in file:
			continue
		elif ".png" in file: #file is a png 
			X.append(split(file)[0])
			Y.append(split(file)[1])
		else: #file is actually a folder with pngs
			for f2 in os.listdir(folder + file):
				X.append(split(file + '/' + f2,file)[0])
				Y.append(split(file + '/' + f2,file)[1])

	#shuffle X and Y together
	c = list(zip(X,Y))
	random.shuffle(c)
	X,Y = zip(*c)

	X = np.reshape(np.array(X),(-1,HEIGHT,WIDTH))
	Y = one_hot(Y).reshape(X.shape[0],-1)

	return X,Y


def reset_weights(model):
	session = backend.get_session()
	for layer in model.layers: 
		if hasattr(layer, 'kernel_initializer'):
			layer.kernel.initializer.run(session=session)

def compile_model():
	conv_model = models.Sequential()
	conv_model.add(layers.Conv1D(32, (3), activation='relu',
	                        input_shape=(HEIGHT,WIDTH)))
	conv_model.add(layers.MaxPooling1D((2)))

	conv_model.add(layers.Conv1D(128, (3), activation='relu'))
	conv_model.add(layers.MaxPooling1D((2)))

	conv_model.add(layers.Flatten())
	conv_model.add(layers.Dropout(0.5))
	conv_model.add(layers.Dense(512, activation='relu'))
	conv_model.add(layers.Dense(10, activation='softmax'))

	LEARNING_RATE = 1e-4

	conv_model.compile(loss='categorical_crossentropy',
	              optimizer = optimizers.Adam(learning_rate=LEARNING_RATE),
	              metrics=['acc'])
	reset_weights(conv_model)

	return conv_model

model = compile_model()
X,Y = get_images()

print(Y)
history_conv = model.fit(X, Y, 
	                            validation_split=0.1, 
	                            epochs=10, 
	                            batch_size=8)

plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper left')
plt.show()

plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy (%)')
plt.xlabel('epoch')
plt.legend(['train accuracy', 'val accuracy'], loc='upper left')
plt.show()


model.save("/home/fizzer/ros_ws/src/controller/src/Driver/driver")
