import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

'''Image directories'''
DATADIR = 'images/training_data'
TESTDIR = os.getcwd() + '/images/test'
'''Get list of categories'''
CATEGORIES = []

directories = os.listdir(os.getcwd() + '/' + DATADIR)

for d in directories:
    if "DS_Store" not in d:
        CATEGORIES.append(d)
CATEGORIES.sort()
print(CATEGORIES)

IMG_SIZE = 224

def prepare(filepath):
    img_array = cv2.imread(filepath, 1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

''' Load model '''
model = tf.keras.models.load_model('modelbinary_crossentropy6-cv-128-nd-5-ds.h5')

i=0

for img in os.listdir(TESTDIR):
    i+=1
    if i == 15:
        break

    path = os.path.join(TESTDIR, img)

    if "DS_Store" not in path:
        print("predicting")
        '''Get image array'''
        predictImage = prepare(path)
        '''Predict always pass list'''
        predict = model.predict([predictImage])
        prediction = CATEGORIES[int(predict[0][0])]
        print('Most probably: ' + prediction)

        ''' Plotting image with prediction '''
        img_array = cv2.imread(path, 1)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        plt.grid(False)
        plt.imshow(new_array)
        plt.xlabel("Prediction: "+prediction)
        plt.show()
