# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet169
from keras.applications.resnet50 import ResNet50
import os,numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,BatchNormalization
from keras import optimizers,callbacks #,regularizers
os.chdir('E:\\Data Science\\deeplearning\\Python scripts')
import utils,pandas as pd

#VGG16 Bottlenck features extraction
VGG16_conv_base = VGG16(include_top=False,weights='imagenet',input_shape=(150,150,3))
VGG16_conv_base.summary()
#Val_acc= 95
#Kaggle score =0.36(small data),0.28(full data)

#InceptionV3 Bottlenck features extraction
InceptionV3_conv_base = InceptionV3(include_top=False,weights='imagenet',input_shape=(150,150,3))
InceptionV3_conv_base.summary()
#Val_acc= 96.5
#Kaggle score =0.344(small data),0.17088(full data)

#ResNet50 Bottlenck features extraction
ResNet50_conv_base = ResNet50(include_top=False,weights='imagenet',input_shape=(350,350,3))
ResNet50_conv_base.summary()
#Val_acc= 60,65
#Kaggle score =0.67(small),0.66(full)

#DenseNet169 Bottlenck features extraction
DenseNet169_conv_base = DenseNet169(include_top=False,weights='imagenet',input_shape=(221,221,3))
DenseNet169_conv_base.summary()
#Val_acc= ,0.15
#Kaggle score =(small),0.985(full)

#Ensemble

#Soft ensemble : 0.13
#Hard ensemble : 0.100

small_data_dir = 'E:\\Data Science\\Data\\CatsVsDogs\\'
full_data_dir = 'E:\\Data Science\\Data\\CatsVsDogs\\'
test_actual ='E:\\Data Science\\Data\\CatsVsDogs\\'
train_dir = os.path.join(full_data_dir,'train')
validation_dir = os.path.join(full_data_dir,'validation')
test_dir = os.path.join(full_data_dir,'test')

data_gen = ImageDataGenerator(rescale=1.0/255)

img_width, img_height = 350, 350
epochs = 20
batchsize = 40

os.chdir('E:\\Data Science\\Data\\CatsVsDogs')

def extract_bottleneck_features(directory,sample_count):
#    features = np.zeros(shape=(sample_count,4,4,512))
    generator = data_gen.flow_from_directory(directory,
                                             target_size=(img_width,img_height),
                                             class_mode=None,
                                             batch_size=batchsize,
                                             shuffle=False
                                             )
    features = ResNet50_conv_base.predict_generator(generator,
                                                 steps=sample_count/batchsize)
    
    return features


os.chdir('E:\\Data Science\\Data\\CatsVsDogs\\')  
train_features = extract_bottleneck_features(train_dir,20000)
np.save(open('ResNet50_bottleneck_features_train_full.npy', 'wb'),train_features)
train_features = np.load(open('ResNet50_bottleneck_features_train_full.npy','rb'))
train_labels = np.array([0] * (20000 // 2) + [1] * (20000 // 2))
y_train = np_utils.to_categorical(train_labels)

validation_features = extract_bottleneck_features(validation_dir,5000)
np.save(open('ResNet50_bottleneck_features_validation_full.npy', 'wb'),validation_features)
#validation_features = np.load(open('ResNet50_bottleneck_features_train_full.npy','rb'))
validation_labels = np.array([0] * (5000 // 2) + [1] * (5000 // 2))
y_validation = np_utils.to_categorical(validation_labels)

test_features = extract_bottleneck_features(test_dir,12500)
np.save(open('ResNet50_bottleneck_features_test_full.npy', 'wb'),test_features)
#test_features = np.load(open('ResNet50_bottleneck_features_test_full.npy','rb'))

#Fully connected layers
model = Sequential()
model.add(Flatten(input_shape=(test_features.shape[1:])))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(512,activation='relu')) 
model.add(Dropout(0.5))   
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))    
model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

model_path = 'E:\\Data Science\\Data\\CatsVsDogs\\DenseNet169_bottlenck_model_full.h5'
model.load_weights(model_path)

# Callbacks
save_weights = callbacks.ModelCheckpoint('DenseNet169_bottlenck_model_full.h5', monitor='val_acc', save_best_only=True)
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.005)

# Train the model

history = model.fit(train_features,y_train,
                    epochs=epochs,
                    batch_size=batchsize,
                    validation_data=(validation_features, y_validation),
                    callbacks=[save_weights, learning_rate_reduction])

utils.plot_loss_accuracy(history)

# Predit the probabilities using differnt bottleneck models and ensemble.

probabilities_InceptionV3 = model.predict_proba(test_features)
probabilities_VGG16 = model.predict_proba(test_features)
probabilities_DenseNet = model.predict_proba(test_features)

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=20,
        class_mode=None,
        shuffle=False)

#Ensembles (Either Use soft of hard combincation of multiple pre-trained nets)
probabilities_ensemble =0.35*probabilities_InceptionV3+0.2*probabilities_VGG16+0.45*probabilities_DenseNet

test_generator = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
            test_dir,
            target_size=(img_width, img_height),
            batch_size=batchsize,
            class_mode=None,
            shuffle=False)
mapper = {}
i = 0
for file in test_generator.filenames:
    id = int(file.split('\\')[1].split('.')[0])
    mapper[id] = round(probabilities_DenseNet[i][1],3)
    i += 1   
tmp = pd.DataFrame({'id':list(mapper.keys()),'label':list(mapper.values())})
os.chdir('E:\\Data Science\\Data\\CatsVsDogs\\')    
tmp.to_csv('submissionDenNet_BN.csv', columns=['id','label'], index=False)
