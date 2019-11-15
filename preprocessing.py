import pandas as pd
import os
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tqdm import tqdm
import pickle
#nltk.download('wordnet')


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', type = str, help = 'dataset name', default = 'dataset')
args = parser.parse_args()


DATA_PATH = './ag_news_csv'

train_data = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'), names=['class', 'title', 'desc'])
test_data = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'), names=['class', 'title', 'desc'])
#print(train_data.head())


def tokenize(text):
    text = text.replace('\\', ' ').lower()
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    stop = stopwords.words('english')
    tokens = [_ for _ in tokens if (_ not in stop) and (len(_) > 2)]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def ngram(tokens, N=2):
    ngram = list()
    for i in range(len(tokens) - (N - 1)):
        ngram.append(" ".join(tokens[i:i + N]))
    return ngram


train_X, train_Y = [], []
test_X, test_Y = [], []

for i, row in tqdm(train_data.iterrows()):
    tokens = tokenize(row.title) + ngram(tokenize(row.title)) + tokenize(row.desc) + ngram(tokenize(row.desc))
    train_X.append(tokens)
    train_Y.append(row['class'])

dataDict = {'train_X': train_X, 'train_Y': train_Y, 'test_X': test_X, 'test_Y': train_Y}

file_name = args.data_name + '.pickle'
pickle.dump(dataDict, open(os.path.join('data', file_name), 'wb'))