import string
import random
from random import randint
import cv2
import numpy as np
import os
import re

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

def one_hot(Y):
	Y_encoded = []

	for p in Y:
		encoded_plate = []     #array to hold 4 one_hot encodings (for 1 plate)
		for i,char in enumerate(p):
			encoding = np.zeros(36,)         # array to hold one_hot encoding for 1 character
			if ord(char) <= 90 and ord(char) >= 65:
				encoding[ord(char)-65] = 1
			else:
				encoding[int(char)+26] = 1

			encoded_plate.append(encoding)
			Y_encoded.append(np.array(encoded_plate))

	return np.asarray(Y_encoded)


  # get the characters from each image, store the png file as well as the correct characters.
def get_characters(p,use='train'):
	if use == 'train':
		img = cv2.imread("/home/fizzer/Pictures/130x30/" + p)
	else:
		img = cv2.imread("/home/fizzer/Pictures/" + p)

	img = cv2.resize(img,(130,30))
	img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	im1 = np.asarray(img[:30,2:28])
	im2 = np.asarray(img[:30,29:55])
	im3 = np.asarray(img[:30,79:105])
	im4 = np.asarray(img[:30,104:130])

	c1 = p[0]
	c2 = p[1]
	c3 = p[2]
	c4 = p[3]

	return [im1,im2,im3,im4], [c1,c2,c3,c4]

def get_images(folder = '/home/fizzer/Pictures/130x30/'):
	plates = os.listdir(folder)
	random.shuffle(plates)

	# Input arrays to CNN
	X = []
	Y = []

	for p in plates:
		X.extend((get_characters(p)[0]))
		Y.extend(get_characters(p)[1])

	X = np.reshape(np.array(X),(-1,30,26))

	#X = np.asarray(temp)
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
	                        input_shape=(30,26)))
	conv_model.add(layers.MaxPooling1D((2)))

	conv_model.add(layers.Conv1D(128, (3), activation='relu'))
	conv_model.add(layers.MaxPooling1D((2)))

	conv_model.add(layers.Conv1D(1024, (3), activation='relu'))
	conv_model.add(layers.MaxPooling1D((2)))


	conv_model.add(layers.Flatten())
	conv_model.add(layers.Dropout(0.5))
	conv_model.add(layers.Dense(512, activation='relu'))
	conv_model.add(layers.Dense(36, activation='softmax'))

	LEARNING_RATE = 0.5e-3

	conv_model.compile(loss='categorical_crossentropy',
	              optimizer=optimizers.RMSprop(lr=LEARNING_RATE),
	              metrics=['acc'])

	reset_weights(conv_model)

	return conv_model


model = compile_model()
X,Y = get_images()
history_conv = model.fit(X, Y, 
	                            validation_split=0.1, 
	                            epochs=10, 
	                            batch_size=8)
model.save("/home/fizzer/ros_ws/src/controller/src/Plate_reader/classifier")
