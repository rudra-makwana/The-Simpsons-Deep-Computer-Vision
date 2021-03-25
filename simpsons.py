import cv2 as cv
import numpy as np
import os
import caer
import canaro
import gc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler
import sklearn.model_selection as skm

IMG_SIZE = (80,80)
channels = 1
char_path = r'C:\Users\rudra\Desktop\Machine Learning\Deep Computer Vision - The Simposons\Simpsons Data\simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path, char)))


char_dict = caer.sort_dict(char_dict, descending = True)

characters = []

count = 0
for i in char_dict:
    count += 1
    characters.append(i[0])
    if count>9:
        break

train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True, verbose=0)

featureSet, labels = caer.sep_train(train, IMG_SIZE = IMG_SIZE)

featureSet = caer.normalize(featureSet)
labels = to_categorical(labels, len(characters))


split_data = skm.train_test_split(featureSet, labels, test_size=.2)
x_train, x_val, y_train, y_val = (np.array(item) for item in split_data)

del featureSet
del labels
del train
gc.collect()

BATCH_SIZE = 32
EPOCHS = 10

datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)

model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim= len(characters),
                                         loss='binary_crossentropy',
                                         decay=1e-6, learning_rate=0.001,
                                         momentum=0.9, nesterov=True)

callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen, steps_per_epoch=len(x_train)//BATCH_SIZE, epochs=EPOCHS, validation_data=(x_val, y_val), validation_steps=len(y_val)//BATCH_SIZE, callbacks=callbacks_list)

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img

test_path = r'C:\Users\rudra\Desktop\Machine Learning\Deep Computer Vision - The Simposons\Simpsons Data\kaggle_simpson_testset\kaggle_simpson_testset\charles_montgomery_burns_10.jpg'

img = cv.imread(test_path)
img = prepare(img)

prediction = model.predict(img)

print(characters[np.argmax(prediction[0])])