import glob
import pandas as pd
import cv2
import numpy as np
import os
from random import shuffle
from csv import writer
import csv
import Augmentor as aug
from keras_preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers
from keras.models import Sequential
from keras.optimizers import SGD
import tflearn
import openpyxl as oxl
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from PIL import Image

TRAIN_DIR = r'C:\Users\Lenovo\Downloads\augmented'
TEST_DIR = r'C:\Users\Lenovo\Downloads\NN Dataset\Test'
TRAIN_DIR2 = r'C:\Users\Lenovo\Downloads\NN Dataset\Train'
IMG_SIZE = 104
LR = 0.01
epoch = 70
MODEL_NAME = 'Sportss'

def create_label(image_name):
    """ Create a one-hot encoded vector from image name """
    word_label = image_name.split('_')[0]
    if word_label == 'Basketball':
        return np.array([1, 0, 0, 0, 0, 0])
    elif word_label == 'Football':
        return np.array([0, 1, 0, 0, 0, 0])
    elif word_label == 'Rowing':
        return np.array([0, 0, 1, 0, 0, 0])
    elif word_label == 'Swimming':
        return np.array([0, 0, 0, 1, 0, 0])
    elif word_label == 'Tennis':
        return np.array([0, 0, 0, 0, 1, 0])
    elif word_label == 'Yoga':
        return np.array([0, 0, 0, 0, 0, 1])

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), create_label(img)])
    shuffle(training_data)
    np.save('trainn_data.npy', training_data)
    return training_data
def validationdata():
    training_data2 = []
    for img in tqdm(os.listdir(TRAIN_DIR2)):
        path = os.path.join(TRAIN_DIR2, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data2.append([np.array(img_data), create_label(img)])
    shuffle(training_data2)
    np.save('vall_data.npy', training_data2)
    return training_data2
def create_test_data():
    testing_data = []

    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_data = cv2.imread(path, 0)
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img_data), create_label(img)])

    shuffle(testing_data)
    np.save('testt_data.npy', testing_data)
    return testing_data


if (os.path.exists('trainn_data.npy')): # If you have already created the dataset:
    train_data = np.load('trainn_data.npy', allow_pickle=True)
    # train_data = create_train_data()
else:# If dataset is not created:
    train_data = create_train_data()

#print(train_data.shape)
if (os.path.exists('testt_data.npy')):
    test_data = np.load('testt_data.npy', allow_pickle=True)
else:
    test_data = create_test_data()
if (os.path.exists('vall_data.npy')):
    z = np.load('vall_data.npy', allow_pickle=True)
else:
    z=validationdata()

opt = SGD(lr=0.001)
# print(train_data.size)
# print(test_data.size)
val =z
vx_train = np.array([i[0] for i in z]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
vy_train = np.array([i[1] for i in z])
train = train_data
test = test_data
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = np.array([i[1] for i in train])

X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]
X1, valid_data = train_test_split(vx_train, random_state=100, test_size=0.3, shuffle=True)
X2, valid_targets = train_test_split(vy_train, random_state=100, test_size=0.3, shuffle=True)

tf.compat.v1.reset_default_graph()

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow(X_train, y_train, batch_size=300)
val_generator = val_datagen.flow(valid_data, valid_targets, batch_size=300)
model = tf.keras.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(50,50,1)))
# model.add(layers.Conv2D(32, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.Conv2D(128, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
# model.add(layers.Conv2D(256, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))  #for normalization
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(6, activation='softmax'))

model.add(layers.Conv2D(input_shape=(50,50,1),filters=64,kernel_size=(3,3), activation="relu"))
model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Dropout(0.75))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(layers.Flatten())

model.add(layers.Dense(units=4096,activation="relu"))
model.add(layers.Dense(units=4096,activation="relu"))
model.add(layers.Dense(units=6, activation="softmax"))



if (os.path.exists('modell.tfl.meta')):
    model.load('./modell.tfl')
else:
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])

    model.fit(train_generator, epochs=30, validation_data=val_generator)
              # steps_per_epoch=len(train_generator)//300,
                        # validation_steps=len(val_generator)//300,
                        # verbose = 1)
    model.save('modell.tfl')

images=[]
label=[]
for i in glob.glob(r'C:\Users\Lenovo\Downloads\NN Dataset\Test\*'):
    img = cv2.imread(i, 0)
    test_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    test_img = test_img.reshape(-1,IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict([test_img])[0]
    pred = np.argmax(prediction)
    images.append(i.split('Test\\')[-1])
    label.append(pred)
out = pd.DataFrame({'image_name': images,'label': label})
out.to_csv(r'C:\Users\Lenovo\OneDrive\Documents\outputs.csv', index=False)
# prediction = model.predict(X_test)
# # print(prediction)
# classes = []
# for x in range(len(prediction)):
#     classn = np.argmax(prediction[x])
#     classes.append(classn)
#
# # plt.show()
# names = create_test_data()[1]
# file = open('outputCNN.csv', "w", newline='')
# writer = csv.writer(file)
# writer.writerow(["image_name", "label"])
# for w in range(len(X_test)):
#     writer.writerow([names[w], classes[w]])
# file.close()
#
# # for w in range(len(testing_data)):
# #     name = str(w) + ".jpg"
# #     names.append(name)
# # c = list(zip(names, testing_data))
#C:\Users\sts\PycharmProjects\projectNN\Test\0.jpg
