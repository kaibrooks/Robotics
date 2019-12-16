import tensorflow as tf
import matplotlib.pyplot as plt
import getThatData
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np


#this is importing a known data set for testing(hand written numbers 0-9)
def pullIt(self):
	#mnist = tf.keras.datasets.mnist
	#(self.x_train, self.y_train),(self.x_test, self.y_test) = mnist.load_data()
	(np.array([N_train, 3, w, h]),np.array([N_train, 3])), ([N_test, 3, w, h], [N_test, 3]) = getThatData.gather_data_and_groundTruth(r"../documentation/images/robot_pix")
	self.x_train = tf.keras.utils.normalize(self.x_train, axis=1)
	self.x_test = tf.keras.utils.normalize(self.x_test, axis=1)


def buildIt(self, classes = 3):
	#this is a sequential model with basic layers
	self.model = Sequential()
	self.model.add(Flatten())				#squash image to row
	self.model.add(Dense(128, activation=tf.nn.relu))
	self.model.add(Dense(128, activation=tf.nn.relu))
	self.model.add(Dense(classes, activation=tf.nn.softmax))
	self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def makeIt(self):
	pullIt(self)
	buildIt(self)


def trainIt(self, eps):
	#training, epoch is how many tests to do, more is better, but too many can be bad as well.
	self.model.fit(self.x_train, self.y_train, epochs=int(eps))
	#evalutating your training
	val_loss, val_acc = model.evaluate(self.x_test, self.y_test)
	print("loss value: " + str(val_loss))
	print("accuracy: "+ str(val_acc))

def proveIt(self,num):
	#this is what your model thinks x_test is
	predictions = model.predict(self.x_test)
	print("This is what I think it is: " + str(np.argmax(predictions[int(num)])))
	#see what x_test[0] actually is
	plt.imshow(self.x_test[int(num)],cmap=plt.cm.binary)
	plt.show()

def runIt(self):
	makeIt(self)
	trainIt(self,3)
	proveIt(self,0)
