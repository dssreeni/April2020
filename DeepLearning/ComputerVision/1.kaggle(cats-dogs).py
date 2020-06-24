# -*- coding: utf-8 -*-
"""

@author: Sreenivas.J
"""
import os
import pandas as pd
from keras.models import Sequential #For FFNN (Dense Layers)

#CNN based models require Conv2D, MaxPooling2D
from keras.layers import Conv2D, MaxPooling2D #For Featuere Extraction
from keras.preprocessing.image import ImageDataGenerator

#FFNN requires Dense, Flatten
from keras.layers import Dense, Flatten 
#from keras import backend as K
os.chdir('E:\\Data Science\\deeplearning\\Python scripts\\kaggle-cats vs dogs')
import utils
#Early stopping is required when system realizes that there is no improvement after ceratin epochs
from keras.callbacks import ModelCheckpoint, EarlyStopping
#Pip install pillow
#import PIL.Image+
os.getcwd()

#Prepare small/full data set by calling Utils class method
train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples = \
                    utils.preapare_small_dataset_for_flow(
                            train_dir_original='E:\\Data Science\\Data\\CatsVsDogs\\train', 
                            test_dir_original='E:\\Data Science\\Data\\CatsVsDogs\\test',
                            target_base_dir='E:\\Data Science\\Data\\CatsVsDogs\\target base dir')

#Convert all images to standard width and height
img_width, img_height = 150, 150
epochs = 2 #30
batch_size = 20

# =============================================================================
# #Channels first for NON Tensorflow/Keras
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else: #For Tensorflow the imput shape is different
#     input_shape = (img_width, img_height, 3)
# =============================================================================
   
ipshape = (img_width, img_height, 3)

#Model begins here
model = Sequential()
#CNN model with 32 filters, filter size of 3X3 and stride as 1 and padding as 0
#Extracts Very high level features
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=ipshape))
model.add(MaxPooling2D((2, 2)))

#2nd level CNN - #Extracts high-Mid level features
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

#3rd level CNN #Extracts Mid-low level features
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

#Flatten all features extracted by multiple CNN layers
model.add(Flatten())

#So far, features got extracted by using CNN and Max Pooling

#Next apply FFNN/FCNN to classify
model.add(Dense(512, activation='relu'))
#2 outputs (Cats Vs Dogs)
#softmax normalizes the probability Bcoz in the Sigmoid the sum of probabilities may NOT become Zero.
#Softmax ensures the sum(probablity) must be Equals to Zero.
#For example: Image1 one outcome might have came as .7 probability of cat and .2 as dog. And the sum is not = 1.
#Hence Softmax normalizes and make the total probability to One
model.add(Dense(2, activation='softmax'))
print(model.summary())

# =============================================================================
# for layer in model.layers:
#     if 'çonv'not in layer.name:
#         continue
# 
# filters, biases = layer.get_weights()
# print(filter.shape)
# =============================================================================
#print(layer.names, filter.shape)

model.compile(loss='binary_crossentropy', 
              optimizer='sgd', #adam #'sgd' means Stochastic GD
              metrics=['accuracy'])

#Scaling of images from 0-255. Standardized image Aspect Ration/intensity/Resolution
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

#DirectoryIterator - Iterate thru the directory
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
print(train_generator.filenames)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
#print(validation_generator.filenames)

#If there is no change in continuos 3 epoch(patience=3: Be patient for 3 epochs), then stop early. Heuristically 3-5 is ideal.
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
#Save weights in model.h5
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(train_generator, steps_per_epoch=2000//batch_size, epochs=2, 
                              validation_data=validation_generator, 
                              validation_steps=1000//batch_size,
                              callbacks=[early_stopping, save_weights])
os.getcwd()


print(history.shape)
#Add both accuracies and losses into historyDataFrame
#historydf = pd.DataFrame(history.history, index=history.epoch)
utils.plot_loss_accuracy(history)

#Now let's apply our model onto Test data
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#print(test_generator.filenames)
probabilities = model.predict_generator(test_generator, nb_test_samples//(batch_size))
#probabilities = model.predict_generator(test_generator, nb_test_samples//(batch_size-5))

mapper = {}
i = 0
# =============================================================================
# tstfile =  'images\\10892.jpg'
# tstfile.split('\\')[1].split('.')[1]
# =============================================================================
for file in test_generator.filenames:
    #print(test_generator.filenames)
    id = int(file.split('\\')[1].split('.')[0])
    #print(file.split('\\')[1].split('.')[0])
    #Lexographic order
    mapper[id] = probabilities[i][1] #Dogs probability
    print(probabilities[i])
    print(i)
    print(id)
    print(mapper[id])
    i += 1
os.getcwd()
#od = collections.OrderedDict(sorted(mapper.items()))  
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
tmp.to_csv('submission_DC.csv', columns=['id','label'], index=False)
