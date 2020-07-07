# -*- coding: utf-8 -*-
"""
Author: Sreeni Jilla
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#CNN based models require Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D
#FFNN requires Flatten, Dense
from keras.layers import Flatten, Dense
from keras import backend as K

#import collections
import os
import pandas as pd
os.chdir("E:/Data Science/deeplearning/Python scripts/Distracted Driver Detection")
import utils_Distracted_Driver
os.chdir("E:/Data Science/Data/Distracted Driver")

#Early stopping is required when systme realizes that there is no improvement after ceratin epochs
from keras.callbacks import ModelCheckpoint, EarlyStopping
#import PIL.Image
os.getcwd()

train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                    utils_Distracted_Driver.preapare_full_dataset_for_flow(
                            train_dir_original='E:/Data Science/Data/Distracted Driver/train', 
                            test_dir_original='E:/Data Science/Data/Distracted Driver/test',
                            target_base_dir='E:/Data Science/Data/Distracted Driver/target base dir')

#Convert all images to standard width and height
img_width, img_height = 150, 150
epochs = 2
batch_size = 20

#Channels first is for Tensorflow for input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else: #For non Tensorflow the imput shape is different
    input_shape = (img_width, img_height, 3)

model = Sequential()
#CNN modedl with 32 filter and the filter size of 3X3 and stride as 1 and padding as 0
model.add(Conv2D(128, (3, 3), activation='relu',input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))

#2nd level CNN
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

#3rd levle CNN
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

#4th level of CNN
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
#So fat fetures got extracted by using CNN and Max Pooling

#Next apply FFNN to classify
model.add(Dense(512, activation='relu'))
#2 outputs (Cats Vs Daogs)
#softmax normalizes the probability Bcoz in the Sigmoid the probability may NOT become 1.
#Softmax ensures the sum(probablity) must be 1.
model.add(Dense(10, activation='softmax'))
print(model.summary())
#-------------------------
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

#Scaling of images from 0-255. Redcue image intensity
train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
# =============================================================================
#     save_to_dir='E:\\Data-Science\\Data\\Distracted Driver\\target base dir\\train_resized')
# =============================================================================
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
# =============================================================================
#     save_to_dir='E:\\Data-Science\\Data\\Distracted Driver\\target base dir\\val_resized')
# =============================================================================

#If there is no change in continuos 3 epoch(patience=3: Be patient for 3 epochs), then stop early. Heuristically 3-5 is ideal.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
#Save weights in model.h5
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(train_generator, steps_per_epoch=nb_train_samples//batch_size, epochs=epochs, validation_data=validation_generator, 
    validation_steps=1000//batch_size,
    callbacks=[early_stopping, save_weights])

#Add both accuracies and losses into historyDataFrame
historydf = pd.DataFrame(history.history, index=history.epoch)

utils_Distracted_Driver.plot_loss_accuracy(history)

#Now let's apply our model onto Test data
test_datagen = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#print(test_generator.filenames)
probabilities = model.predict_generator(test_generator,nb_test_samples/(batch_size))
#probabilities = model.predict_generator(test_generator, nb_test_samples//(batch_size-5))
type(probabilities)
 
headers=['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
df = pd.DataFrame(probabilities,columns=headers)


#probabilities = (np.rint(probabilities)).astype(int)

mapper = []
i = 0
for file in test_generator.filenames:
    id = int(file.split('_')[1].split('.')[0])
    #print(id)
    mapper.append(id)
    i += 1
    
df['img'] = mapper

df = df.reindex_axis(['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9'],axis = 1)
#od = collections.OrderedDict(sorted(mapper.items()))    
#tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})
df.sort_values(by=['img'],inplace=True)
df['img'] = 'img_'+df['img'].astype(str)+'.jpg'
os.getcwd()
df.to_csv('submission4.csv', index=False)
