import os
import shutil
import sys
# organize dataset into a useful structure
from os import listdir
from random import random
from random import seed
from shutil import copyfile

import pandas as pd
import requests
from bs4 import BeautifulSoup
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

genres_list = ['comedy', 'sci-fi', 'horror', 'romance',
               'action', 'thriller', 'drama', 'mystery',
               'crime', 'animation', 'adventure', 'fantasy']


# SCRAPING
def all_genres(genres, amount, name):
    """
    Apply scraper on defined list of genres
    :param genres: list
    :param amount: int, how many pages should be scraped per genre
    :param name: str, name of csv file
    :return:
    """
    for z, c in tqdm(enumerate(genres)):
        print(c)
        get_thumbnail_link(c, f"{name}.csv", amount)
def get_thumbnail_link(genre, csv_name, amount):
    """
    Scrapes links to images, rating, name of movies.
    :param genre: str
    :param csv_name: str
    :param amount: int
    :return: thumbnails.csv (specify yourself)
    """
    i = 1
    header = True
    images = []
    nams = []
    ratings = []

    # iterate over pages (50 movies per page are displayed)
    while i < amount:
        try:

            url = f'https://www.imdb.com/search/title/?genres={genre}&start={i}&explore=title_type,genres&ref_=adv_nxt'
            web = requests.get(url)
            soup = BeautifulSoup(web.text, 'html.parser')
            i += 50

            # locate image thumbnail to get image link and name
            imgs = soup.findAll('div', class_="lister-item-image float-left")
            # locate rating
            rating = soup.findAll('div', class_="inline-block ratings-imdb-rating")

            for j in tqdm(range(0, len(imgs))):
                # links
                # replace numbers to get a link to a bigger image
                images.append(
                    imgs[j].a.img['loadlate'].replace('UX67', 'UX182').replace('67', '182').replace('98', '268'))
                # names
                nams.append(imgs[j].a.img['alt'])
                try:
                    # ratings
                    ratings.append(float(rating[j].strong.text))
                except IndexError:
                    ratings.append('none')

                new = pd.DataFrame(data={'Name': nams,
                                         'Image': images,
                                         'Rating': ratings,
                                         'Genre': genre})

                if j % 49 == 0:
                    new.to_csv(csv_name,
                               sep=',',
                               header=header,
                               mode='a',
                               encoding='utf-8')

                    images = []
                    ratings = []
                    nams = []

                header = False


        except (IndexError, KeyboardInterrupt):
            return new
    return new
def delete_headers(df, identifier, name):
    """
    Add unique identifier to data.
    Filter unnecessary headers that got created.
    :param df: df
    :param identifier: df of one column (unique ID's created with uuid)
    :return: data/clean_thumbs.csv
    """
    # read data
    df = pd.read_csv(df)
    identifier = pd.read_csv(identifier)

    # delete headers
    df = df[df['Name'] != 'Name']

    # add identifier
    df['ID'] = identifier['unique_ID'][:len(df)]
    df.to_csv(f'data/{name}.csv', sep=',', encoding='utf-8')

    return df


# DATA PREPARATION
def prepare_data(df):
    """
    Create labels with LabelEncoder. Drop NA's. Shorten ID's.
    :param df: data/clean_thumbs.csv
    :return: label_thumbs.csv
    """
    # read data
    df = pd.read_csv(df)
    df = df.dropna()

    # create labels from genre
    le = preprocessing.LabelEncoder()
    df['label'] = le.fit_transform(df['Genre'])

    df['ID'] = [int(i) for i in df['ID']]
    df['label'] = pd.Categorical(df['label'])

    try:
        df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)
    except:
        return df, df.to_csv("label_thumbs.csv", sep=',', encoding='utf-8')

    return df, df.to_csv("label_thumbs.csv", sep=',', encoding='utf-8')
def split(df):
    """
    Split into train test
    :param df:
    :return:
    """

    df = pd.read_csv(df)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    X = df
    y = df.label.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    X_train['split'] = 'train'
    X_test['split'] = 'test'

    merged_back = X_train.append(X_test)

    return X_train, X_test, y_train, y_test, merged_back
def create_directories():
    # STEP 1
    BASE_DIR = 'images/'

    # STEP 2
    SUB_DIRS = ['train', 'test']

    # STEP 3
    for sub_dir in SUB_DIRS:
        if not os.path.exists(BASE_DIR + sub_dir):
            os.makedirs(BASE_DIR + sub_dir)


# DOWNLOADS
def download_image(image, filename, genre):
    """
    Downloads one image by accessing image link and naming it as genre_ID.
    :param image: str
    :param filename: int
    :param genre: str
    :return:
    """

    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(str(f'images/{genre}_{filename}.jpeg'), 'wb') as f:
            shutil.copyfileobj(r.raw, f)
def download_all(df):
    """
    Download the images by iterating over clean thumbnail link and requesting the image link.
    :param df:df
    :return:
    """
    [download_image(row[0], row[1], row[2]) for row in tqdm(df[['Image', 'ID', 'Genre']].to_numpy())]


# MACHINE LEARNING
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


if __name__ == '__main__':
    # create csv 'thumbnails.csv' with image links and metadata of imdb movies
    print(all_genres(genres_list, 9000, 'thumbnails'))

    # delete extra headers
    delete_headers(df='thumbnails.csv', name='clean_thumbs', identifier='data/un_ID.csv')

    # prepare data
    prepare_data('data/clean_thumbs.csv')

    # split data
    X_train, X_test, y_train, y_test, merged_back = split('data/label_thumbs.csv')

    # create dictionaries to store the images per genre
    create_directories()

    # start download
    print(download_all(merged_back))

    # sort data into right folders and train/test split
    sort_images()

    train_it, test_it = prepare_iterators()
    print(train_model(train_it, test_it))