# -*- coding: utf-8 -*-
"""
Author: Sreeni Jilla
"""

import os
import shutil
import matplotlib.pyplot as plt
import random

def preapare_full_dataset_for_flow(train_dir_original, test_dir_original, target_base_dir, val_percent=0.2):    
# =============================================================================
#     train_dir_original='E:\\Data-Science\\Distracted Driver\\imgs\\train'
#     test_dir_original='E:\\Data-Science\\Distracted Driver\\imgs\\test'
#     target_base_dir='E:\\Data-Science\\Distracted Driver\\target base dir'
#     val_percent=0.2
# =============================================================================
    train_dir = os.path.join(target_base_dir, 'train')
    validation_dir = os.path.join(target_base_dir, 'validation')
    test_dir = os.path.join(target_base_dir, 'test')

    if not os.path.exists(target_base_dir):          
        os.mkdir(target_base_dir)
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        os.mkdir(test_dir)
        for c in range(0,10,1): 
            os.mkdir(os.path.join(train_dir, 'c'+str(c)))
            os.mkdir(os.path.join(validation_dir, 'c'+str(c)))
        os.mkdir(os.path.join(test_dir, 'images'))
        print('created the required directory structure')
        
        train_classes = os.listdir(train_dir_original)
        
        train_files = list()
        for cls in train_classes:
            print(cls)
            dir_path = os.path.join(train_dir_original, cls)
            print(dir_path)
            files = os.listdir(dir_path)
            for f in files:
                train_files.append(os.path.join(dir_path, f))
            
        len(train_files)
        random.shuffle(train_files)
        n = int(len(train_files) * val_percent)
        val = train_files[:n]
        train = train_files[n:]  

        for t in train:
            print(t)
            if 'c0' in t:
                print('0')
                shutil.copy2(t, os.path.join(train_dir, 'c0'))
            elif 'c1' in t:
                print('1')
                shutil.copy2(t, os.path.join(train_dir, 'c1'))
            elif 'c2' in t:
                print('2')
                shutil.copy2(t, os.path.join(train_dir, 'c2'))
            elif 'c3' in t:
                print('3')
                shutil.copy2(t, os.path.join(train_dir, 'c3'))
            elif 'c4' in t:
                print('4')
                shutil.copy2(t, os.path.join(train_dir, 'c4'))
            elif 'c5' in t:
                print('5')
                shutil.copy2(t, os.path.join(train_dir, 'c5'))
            elif 'c6' in t:
                print('6')
                shutil.copy2(t, os.path.join(train_dir, 'c6'))
            elif 'c7' in t:
                print('7')
                shutil.copy2(t, os.path.join(train_dir, 'c7'))
            elif 'c8' in t:
                print('8')
                shutil.copy2(t, os.path.join(train_dir, 'c8'))
            else:
                print('9')
                shutil.copy2(t, os.path.join(train_dir, 'c9'))
     
        for v in val:
            print(v)
            if 'c0' in v:
                print('0')
                shutil.copy2(v, os.path.join(validation_dir, 'c0'))
            elif 'c1' in v:
                print('1')
                shutil.copy2(v, os.path.join(validation_dir, 'c1'))
            elif 'c2' in v:
                print('2')
                shutil.copy2(v, os.path.join(validation_dir, 'c2'))
            elif 'c3' in v:
                print('3')
                shutil.copy2(v, os.path.join(validation_dir, 'c3'))
            elif 'c4' in v:
                print('4')
                shutil.copy2(v, os.path.join(validation_dir, 'c4'))
            elif 'c5' in v:
                print('5')
                shutil.copy2(v, os.path.join(validation_dir, 'c5'))
            elif 'c6' in v:
                print('6')
                shutil.copy2(v, os.path.join(validation_dir, 'c6'))
            elif 'c7' in v:
                print('7')
                shutil.copy2(v, os.path.join(validation_dir, 'c7'))
            elif 'c8' in v:
                print('8')
                shutil.copy2(v, os.path.join(validation_dir, 'c8'))
            else:
                print('9')
                shutil.copy2(v, os.path.join(validation_dir, 'c9'))
                
        files = os.listdir(test_dir_original)
        test_files = [os.path.join(test_dir_original, f) for f in files]
        for t in test_files:
            shutil.copy2(t, os.path.join(test_dir, 'images'))
    else:
        print('required directory structure already exists. learning continues with existing data')

    nb_train_samples = 0
    nb_validation_samples = 0
    for c in range(0,10,1):
        nb_train_samples = nb_train_samples + len(os.listdir(os.path.join(train_dir, 'c'+str(c))))
    print('total training images:', nb_train_samples)
    for c in range(0,10,1):
        nb_validation_samples = nb_validation_samples + len(os.listdir(os.path.join(validation_dir, 'c'+str(c))))
    print('total validation images:', nb_validation_samples)
    nb_test_samples = len(os.listdir(os.path.join(test_dir, 'images')))
    print('total test images:', nb_test_samples )
    
    return train_dir, validation_dir, test_dir, nb_train_samples, nb_validation_samples, nb_test_samples


def plot_loss_accuracy(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(history.epoch))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
