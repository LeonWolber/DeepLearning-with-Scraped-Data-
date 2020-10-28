import os
import sys
from os import listdir
from random import random
from random import seed
from shutil import copyfile

from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

def sort_images():

    # create directories
    new_home = 'images/'
    new_home = 'image_data/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectorie
        labeldirs = list(merged_back.Genre.unique())
        for labldir in labeldirs:
            newdir = new_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)

    # seed random number generator
    seed(1)
    # define ratio of pictures to use for validation
    val_ratio = 0.17
    # copy training dataset images into subdirectories
    src_directory = 'image_data'
    for file in listdir(src_directory):
        src = src_directory + '/' + file
        dst_dir = 'train/'
        if random() < val_ratio:
            dst_dir = 'test/'
        if file.startswith('comedy'):
            dst = new_home + dst_dir + 'comedy/' + file
            copyfile(src, dst)
        elif file.startswith('action'):
            dst = new_home + dst_dir + 'action/' + file
            copyfile(src, dst)
        elif file.startswith('adventure'):
            dst = new_home + dst_dir + 'adventure/' + file
            copyfile(src, dst)
        elif file.startswith('animation'):
            dst = new_home + dst_dir + 'animation/' + file
            copyfile(src, dst)
        elif file.startswith('crime'):
            dst = new_home + dst_dir + 'crime/' + file
            copyfile(src, dst)
        elif file.startswith('drama'):
            dst = new_home + dst_dir + 'drama/' + file
            copyfile(src, dst)
        elif file.startswith('fantasy'):
            dst = new_home + dst_dir + 'fantasy/' + file
            copyfile(src, dst)
        elif file.startswith('horror'):
            dst = new_home + dst_dir + 'horror/' + file
            copyfile(src, dst)
        elif file.startswith('mystery'):
            dst = new_home + dst_dir + 'mystery/' + file
            copyfile(src, dst)
        elif file.startswith('romance'):
            dst = new_home + dst_dir + 'romance/' + file
            copyfile(src, dst)
        elif file.startswith('sci-fi'):
            dst = new_home + dst_dir + 'sci-fi/' + file
            copyfile(src, dst)
        elif file.startswith('thriller'):
            dst = new_home + dst_dir + 'thriller/' + file
        else:
            pass


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(200, 200, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(12, activation='categorical'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()


def prepare_iterators():
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_it = datagen.flow_from_directory('image_data/train/',
                                           class_mode='categorical',
                                           batch_size=64,
                                           target_size=(200, 200))

    test_it = datagen.flow_from_directory('image_data/test/',
                                          class_mode='categorical',
                                          batch_size=64,
                                          target_size=(200, 200))
    return train_it, test_it


def train_model(train, test):
    # define model
    model = define_model()
    print(model.summary())
    # create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    # fit model
    history = model.fit_generator(train, steps_per_epoch=len(train),
        validation_data=test, validation_steps=len(test), epochs=50, verbose=1)
    # evaluate model
    _, acc = model.evaluate_generator(test, steps=len(test), verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)