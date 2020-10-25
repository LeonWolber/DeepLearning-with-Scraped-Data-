import os
import shutil
import pandas as pd
import requests
from bs4 import BeautifulSoup
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