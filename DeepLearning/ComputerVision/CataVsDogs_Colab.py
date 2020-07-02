from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras import backend as K
import os
import pandas as pd
os.chdir('/content') #Colab path

import shutil
import matplotlib.pyplot as plt
import random

from keras.callbacks import ModelCheckpoint, EarlyStopping

#import PIL.Image
os.getcwd()

#Prepare Full Dataset
def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        for c in ['dogs', 'cats']: 
            os.mkdir(os.path.join(train_dir, c))
            os.mkdir(os.path.join(validation_dir, c))
        os.mkdir(os.path.join(test_dir, 'images'))
        print('created the required directory structure')
        
        files = os.listdir(train_dir_original)
        train_files = [os.path.join(train_dir_original, f) for f in files]
        random.shuffle(train_files)    
        n = int(len(train_files) * val_percent)
        val = train_files[:n]
        train = train_files[n:]  

        for t in train:
            if 'cat' in t:
                shutil.copy2(t, os.path.join(train_dir, 'cats'))
            else:
                shutil.copy2(t, os.path.join(train_dir, 'dogs'))
     
        for v in val:
            if 'cat' in v:
                shutil.copy2(v, os.path.join(validation_dir, 'cats'))
            else:
                shutil.copy2(v, os.path.join(validation_dir, 'dogs'))
        files = os.listdir(test_dir_original)
        test_files = [os.path.join(test_dir_original, f) for f in files]
        for t in test_files:
            shutil.copy2(t, os.path.join(test_dir, 'images'))
    else:
        print('required directory structure already exists. learning continues with existing data')

    nb_train_samples = 0  
    nb_validation_samples = 0
    for c in ['dogs', 'cats']:
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, c)))
    print('total training images:', nb_train_samples)
    for c in ['dogs', 'cats']:
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, c)))
    print('total validation images:', nb_validation_samples)
    nb_test_samples = len(os.listdir(os.path.join(test_dir, 'images')))
    print('total test images:', nb_test_samples )
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples
###


train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples,nb_test_samples = \
                            preapare_full_dataset_for_flow(
                            train_dir_original='train',  #Google Colab notation
                            test_dir_original='test',  #Google Colab notation
                            target_base_dir='target base dir')  #Google Colab notation

img_width, img_height = 150, 150
epochs = 5
batch_size = 20

# =============================================================================
# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)
# =============================================================================
  
ipshape = (img_width, img_height, 3)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu',input_shape = ipshape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())

model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

#Data Augmentation (New Data Generation)
#Overfitting can be addressed with Data Augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
#If we want, you can write all these Augmented data into new files
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
    callbacks=[save_weights, early_stopping])

#historydf = pd.DataFrame(history.history, index=history.epoch)
#utils.plot_loss_accuracy(history)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
#print(test_generator.filenames)
probabilities = model.predict_generator(test_generator, nb_test_samples//batch_size)

mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('/')[1].split('.')[0]) #Google Colab notation (backward slash)
    #Lexographic order
    #mapper[id] = probabilities[i][0] #Cats
    mapper[id] = probabilities[i][1] #Dogs
    i += 1
    
#od = collections.OrderedDict(sorted(mapper.items()))    
#tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})    
#tmp.to_csv('submission.csv', columns=['id','label'], index=False)
