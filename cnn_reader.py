import string
import random
from random import randint
import cv2
import numpy as np
import os
import re
from matplotlib import pyplot as plt
# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

def one_hot(Y):
	Y_encoded = []

	for p in Y:
		encoded_plate = []     #array to hold 4 one_hot encodings (for 1 plate)
		for i,char in enumerate(p):
			encoding = np.zeros(36,)         # array to hold one_hot encoding for 1 character
			# print(char)
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
		img = cv2.imread('/home/fizzer/Pictures/Plates_training/Combined/' + p)
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

def get_images(folder = '/home/fizzer/Pictures/Plates_training/Combined/'):
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

def Plot():
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

#One-hot encoding
# test_imgs = !ls "/home/fizzer/Pictures/Plates_training/Post/"
# test_imgs = [re.split('\s |\t',img) for img in test_imgs]
def Confusion():

	test_imgs = os.listdir("/home/fizzer/Pictures/Plates_training/Combined/")
	random.shuffle(test_imgs)

	# tmp = []
	# for element in test_imgs:
	#   tmp = tmp + element
	# test_imgs = tmp
	# # now we have all of the test plates in test_imgs
	model = tensorflow.keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/src/nodes/Plate_reader/classifier4')


	y_true = []
	y_pred = []

	for img in test_imgs:
	  gnd_truth, prediction = test_CNN(img, model)
	  # if gnd_truth != prediction:
	  # 	img = np.asarray(img)
	  # 	cv2.imshow(str(gnd_truth), img)
	  for i in range(len(gnd_truth)):
	    y_pred.append(str(prediction[i]))
	    y_true.append(str(gnd_truth[i]))

	count_error = 0
	count_correct = 0 

	for i,c in enumerate(y_true):
	  if y_pred[i] != c:
	    count_error += 1
	    print('proper: ' + str(c) + ', pred: ' + str(y_pred[i]))
	  else:
	    count_correct += 1

	print(count_error/len(y_true), count_correct/len(y_true))

	label = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

	for c in label:
	  if c not in y_true:
	    print(c)

	conf = confusion_matrix(y_true,y_pred)
	df_cm = pd.DataFrame(conf, index = [i for i in label], columns =[i for i in label])
	plt.figure(figsize = (25,18))
	sn.heatmap(df_cm, annot=True)
	plt.show()

# Display images in the training data set. 
def test_CNN(plate_path, model):

  char_image, char_text = get_characters(plate_path) #sectioned images with individual characters
  #and characters for sectioned images (correct answers)

  y_predict, predicted_char = [],[]

  for i in range(4):
    img = np.expand_dims(char_image[i], axis=0)
    prediction = model.predict(img)[0]  #one-hot encoded output from CNN
    
    idx = np.where(prediction == np.max(prediction))[0] 
    if idx >= 26:
      res = str(int(idx-26))
    else:
      res = chr(idx+65)
    predicted_char.append(res)               #actual character prediction from CNN
    if predicted_char[i] != char_text[i]:
    	randomint = random.randint(0,1000)
    	cv2.imshow(str(predicted_char[i])+','+str(char_text), char_image[i])
  if char_text != predicted_char:
  	print(char_text, predicted_char)

  return (char_text, predicted_char)

model = compile_model()
X,Y = get_images()
history_conv = model.fit(X, Y, 
	                            validation_split=0.1, 
	                            epochs=40, 
	                            batch_size=16)
Plot()
model.save("/home/fizzer/ros_ws/src/controller_pkg/src/nodes/Plate_reader/classifier4")
Confusion()


