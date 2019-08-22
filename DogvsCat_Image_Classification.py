import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
import time
'''
To remove DS_Store files in MacOS
find . -name '.DS_Store' -type f -delete
'''

'''
Reading and writing large pickle files without running into OSError
New pickle_dump and pickle_load methods to be used to create or load a pickle files
'''

class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))




DATADIR = 'images/training_data'
CATEGORIES = []

directories = os.listdir(os.getcwd() + '/' + DATADIR)

for d in directories:
    if "DS_Store" not in d:
         g=d
         CATEGORIES.append(g)

print(CATEGORIES)

IMG_SIZE = 224
# NAME = 'Private_Property_CNN-{}'.format(int(time.time()))
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
training_data = []

'''
Read in images from the specified DATADIR
Make the categories specified in CATEGORIES into numbers
Create the training data array using IMG_SIZE
'''
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #path to category directory
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            if "DS_Store" not in img:
                try:
                    ''' cv2.imread path first parameter and second parameter is color (1) and greyscale (0)'''
                    img_array = cv2.imread(os.path.join(path,img),1)
                    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    training_data.append([new_array, class_num])
                except:
                    pass
'''========================================================================'''

'''
Randomly shuffle the data in training data
Create array X (for the features) and y (for the lables)
Reshape the arrays using numpy change the 3 to 1 for greyscale
    (The 4th parameter in reshape)
Save X and y using pickle
'''
def prepare_data():
    '''shuffle data'''
    random.shuffle(training_data)
    # for sample in training_data[:10]:
    #     print(sample[1])

    X = []
    y = []

    for features, label in training_data:
        X.append(features)
        y.append(label)

    '''reshape X array -1, IMG is the image size and 3 is for colour change to 1 for greyscale'''
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array(y)


    '''Save with pickle'''
    pickle_dump(X,'X.pickle')
    pickle_dump(y,'y.pickle')

'''========================================================================'''
'''
Train the model using different dense layers, layer size and conv layers
save the model
'''

def train_modelCNN(eps, dense_layer, layer_size, conv_layer, loss_function):
    '''load in X and y with pickle'''
    X = pickle_load('X.pickle')
    y = pickle_load('y.pickle')

    ''' divide all values by 255.0 to normalise the values '''
    X = X/255.0

    '''######First Layer########'''
    model = Sequential()
    #Convolutional layer
    model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
    #activation layer
    model.add(Activation('softmax'))
    #model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    '''######Second Layer########'''
    for c in range(conv_layer-1):
        #Convolutional layer
        model.add(Conv2D(layer_size, (3,3)))
        #activation layer
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())    #this converts our 3D feature maps to 1D feature vectors
    '''######Third Layer########'''
    for d in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))
        #model.add(Activation('softmax'))
        model.add(Dropout(0.2))

    '''######Output Layer########'''
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])

                    #Batch size is the amount at a time. Depends on data size
    model.fit(X, y, batch_size=32, epochs=eps, validation_split=0.2, callbacks=[tensorboard]) #20% validation

    '''save the model'''
    model.save('model.h5')
'''========================================================================'''

'''  Read in images and create the training data array ONLY RUN ONCE'''
#create_training_data()
'''Length of training data'''
#print(len(training_data))
'''Prepare data for model and output X and y as pickle files ONLY RUN ONCE'''
#prepare_data()

'''For logs using tensorboard type in terminal MacOS:
    tensorboard --logdir='logs/' ''' ##To get tensorboard

'''
Run the model using different dense, size, conv and epochs
'''

dense_layers = [2]
layer_sizes = [64]
conv_layers = [3]
loss_functions = ['binary_crossentropy']
EPOCHS = 5
for loss_function in loss_functions:
    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:
                NAME = "{}-cv-{}-nd-{}-ds-{}-los-{}".format(conv_layer, layer_size, dense_layer, loss_function, int(time.time()))
                tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME), histogram_freq=0,
                            write_graph=True, write_images=True)
                print(NAME)
                train_modelCNN(EPOCHS, dense_layer, layer_size, conv_layer, loss_function)
