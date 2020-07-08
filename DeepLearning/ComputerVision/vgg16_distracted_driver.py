# -*- coding: utf-8 -*-
"""
Author: Sreeni Jilla
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd
import os
from keras.utils import np_utils

os.chdir("D:/Data Science/deeplearning/Python scripts/Distracted Driver Detection")
import utils_Distracted_Driver
os.chdir("D:/Data Science/Data/Distracted Driver")


img_width, img_height = 150, 150
epochs = 50
batch_size = 20

def save_bottlebeck_features(model, preprocess_fn, train_dir, validation_dir, test_dir, bottleneck_dir):
    if not os.path.exists(bottleneck_dir):
        os.mkdir(bottleneck_dir) 
        os.chdir(bottleneck_dir) 
        datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_fn)

        train_generator = datagen.flow_from_directory(
                 train_dir,
                 target_size=(img_width, img_height),
                 batch_size=batch_size,
                 class_mode=None,
                 shuffle=False)
        bottleneck_features_train = model.predict_generator(
               train_generator, len(train_generator))
        np.save(open('bottleneck_features_train.npy', 'wb'),
               bottleneck_features_train)
 
        validation_generator = datagen.flow_from_directory(
                validation_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_validation = model.predict_generator(
                validation_generator, len(validation_generator))
        np.save(open('bottleneck_features_validation.npy', 'wb'),
                bottleneck_features_validation)
    
    
        test_generator = datagen.flow_from_directory(
                test_dir,
                target_size=(img_width, img_height),
                batch_size=batch_size,
                class_mode=None,
                shuffle=False)
        bottleneck_features_test = model.predict_generator(
                test_generator, len(test_generator))
        np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)
    else:
        print('bottleneck directory already exists')

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    utils_Distracted_Driver.preapare_full_dataset_for_flow(
                            train_dir_original='D:/Data Science/Data/Distracted Driver/train', 
                            test_dir_original='D:/Data Science/Data/Distracted Driver/test',
                            target_base_dir='D:/Data Science/Data/Distracted Driver/bottle_neck_target')

model = applications.VGG16(include_top=False, weights='imagenet', 
                           input_shape=(img_width, img_height, 3))

bottleneck_dir = 'D:/Data Science/Data/Distracted Driver/bottleneck_features'
preprocess = applications.vgg16.preprocess_input
save_bottlebeck_features(model, preprocess, train_dir, validation_dir, test_dir, bottleneck_dir)

if(os.getcwd() == "D:/Data Science/Data/Distracted Driver"):
    os.chdir("D:/Data Science/Data/Distracted Driver/bottleneck_features")

X_train = np.load(open('bottleneck_features_train.npy','rb'))
train_labels = np.array(
        [0] * (nb_train_samples // 10) + [1] * (nb_train_samples // 10)+
        [2] * (nb_train_samples // 10) + [3] * (nb_train_samples // 10)+
        [4] * (nb_train_samples // 10) + [5] * (nb_train_samples // 10)+
        [6] * (nb_train_samples // 10) + [7] * (nb_train_samples // 10)+
        [8] * (nb_train_samples // 10) + [9] * (nb_train_samples // 10))
y_train = np_utils.to_categorical(train_labels)

len(train_labels)
len(X_train)

X_validation = np.load(open('bottleneck_features_validation.npy','rb'))
validation_labels = np.array(
        [0] * (nb_validation_samples // 10) + [1] * (nb_validation_samples // 10)+
        [2] * (nb_validation_samples // 10) + [3] * (nb_validation_samples // 10)+
        [4] * (nb_validation_samples // 10) + [5] * (nb_validation_samples // 10)+
        [6] * (nb_validation_samples // 10) + [7] * (nb_validation_samples // 10)+
        [8] * (nb_validation_samples // 10) + [9] * (nb_validation_samples // 10))
y_validation = np_utils.to_categorical(validation_labels)


len(validation_labels)
len(X_validation)

top_model = Sequential()
# flateen the output of VGG16 model to 2D Numpy matrix (n*D)
top_model.add(Flatten(input_shape=X_train.shape[1:]))
# hidden layer of 256 neurons
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
# the output layer: we have 10 claases
top_model.add(Dense(10, activation='softmax'))

top_model.summary()

top_model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
save_weights = ModelCheckpoint('bottlenck_model.h5', monitor='val_loss', save_best_only=True)

history = top_model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(X_validation, y_validation),
              callbacks=[save_weights, early_stopping])

utils_Distracted_Driver.plot_loss_accuracy(history)


test_data = np.load(open('bottleneck_features_test.npy','rb'))

probabilities = top_model.predict_proba(test_data)

test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

headers=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
df = pd.DataFrame(probabilities,columns=headers)

mapper = []
i = 0
for file in test_generator.filenames:
    id = int(file.split('_')[1].split('.')[0])
    mapper.append(id)
    i += 1
    
df['img'] = mapper

df = df.reindex_axis(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'],axis = 1)
#od = collections.OrderedDict(sorted(mapper.items()))    
#tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})
df.sort_values(by=['img'],inplace=True)
df['img'] = 'img_'+df['img'].astype(str)+'.jpg'
df.to_csv('submission_vgg161.csv', index=False)
