# -*- coding: utf-8 -*-

import argparse
import time, gc
import pandas as pd
from lstm import train_and_test, preprocess, test
import numpy as np
import torch
import os
import pickle
from sklearn.model_selection import train_test_split

def main():
    print("Predict Youtube cross genre")
    directory = 'data/csv/'
    '''df_data, y = preprocess_data(directory, 'train_news_twitter.csv')
    df_test, test_y = preprocess_data(directory, 'youtube_train.csv')
    train_and_test(df_data, y, df_test, test_y, 100, 'youtube')

    print("Predict News cross genre")
    directory = 'data/csv/'
    df_data, y = preprocess_data(directory, 'train_youtube_twitter.csv')
    df_test, test_y = preprocess_data(directory, 'news_train.csv')
    train_and_test(df_data, y, df_test, test_y, 100, 'news')'''

    print("Predict Twitter cross genre")
    #directory = 'data/csv/'
    #df_data, y = preprocess_data(directory, 'twitter_train.csv')
    #df_test, test_y = preprocess_data(directory, 'twitter_train.csv')
    #print("Shape of train and test: ", df_data.shape, df_test.shape)
    #train_and_test(df_data, y, df_test, test_y, 100, 'twitter')

    '''directory = 'data/csv/'
    df_data, y, df_test, test_y = preprocess_data(directory, 'surprise_test.csv', split=True)
    print("Shape of train and test: ", df_data.shape, df_test.shape)
    train_and_test(df_data, y, df_test, test_y, 100, 'surprise')'''

    #cross genre

    '''model = 'models/news_model_cg_0.557.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/news_corpus_cg_0.557.pk', 'rb'))
    corpus.batch_size = 16
    model.batch_size = 16
    df_test, test_y = preprocess_data(directory, 'twitter_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_CROSS_twitter_2', test=False)

    model = 'models/news_model_cg_0.557.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/news_corpus_cg_0.557.pk', 'rb'))
    corpus.batch_size = 10
    model.batch_size = 10
    df_test, test_y = preprocess_data(directory, 'news_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_CROSS_news_1', test=False)

    model = 'models/youtube_model_cg_0.558.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/youtube_corpus_cg_0.558.pk', 'rb'))
    corpus.batch_size = 2
    model.batch_size = 2
    df_test, test_y = preprocess_data(directory, 'youtube_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_CROSS_youtube_1', test=False)'''

    '''model = 'models/news_model_in.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/news_corpus_in.pk', 'rb'))
    corpus.batch_size = 1
    model.batch_size = 1
    df_test, test_y = preprocess_data(directory, 'surprise_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_CROSS_kb_1', test=False)'''

    #in_genre

    model = 'models/youtube_model_in.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/youtube_corpus_in.pk', 'rb'))
    corpus.batch_size = 2
    model.batch_size = 2
    df_test, test_y = preprocess_data(directory, 'youtube_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_IN_youtube_1', test=False)

    '''model = 'models/news_model_in.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/news_corpus_in.pk', 'rb'))
    corpus.batch_size = 10
    model.batch_size = 10
    df_test, test_y = preprocess_data(directory, 'news_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_IN_news_1', test=False)

    model = 'models/twitter_model_in.pt'
    model = torch.load(model)
    corpus = pickle.load(open('models/twitter_corpus_in.pk', 'rb'))
    corpus.batch_size = 16
    model.batch_size = 16
    df_test, test_y = preprocess_data(directory, 'twitter_test.csv', predict=True)
    test(df_test, test_y, model, corpus, 'IJS-KD_IN_twitter_1', test=False)'''



def preprocess_data(directory, input_file, delimiter="\t", predict=False, split=False):
    # uncomment this to read  data from csv
    data_iterator = pd.read_csv(directory + input_file, encoding="utf-8", delimiter=delimiter, chunksize=1000)
    df_data = pd.DataFrame()
    for sub_data in data_iterator:
        df_data = pd.concat([df_data, sub_data], axis=0)
        gc.collect()
    print("Data shape before preprocessing:", df_data.shape)
    #df_data = df_data[:100]

    df_data = preprocess(df_data)
    df_data.to_csv(directory + "data_preprocessed.csv", encoding="utf8", sep="\t", index=False)

    print(df_data.columns.tolist())

    # shuffle the corpus and optionaly choose the chunk you want to use if you don't want to use the whole thing - will be much faster
    df_data = df_data.sample(frac=1, random_state=1)

    print("Data shape: ", df_data.shape)

    if split:
        df_train, df_test = train_test_split(df_data, test_size=0.1)
        tags = df_train.gender
        m_data = df_train[df_train['gender'] == 'M']
        f_data = df_train[df_train['gender'] == 'F']
        print('Males: ', m_data.shape, 'Females: ', f_data.shape)
        df_train = df_train.drop(['gender'], axis=1)
        y_train = np.array([0 if tmp_y=='M' else 1 for tmp_y in tags])

        tags = df_test.gender
        m_data = df_test[df_test['gender'] == 'M']
        f_data = df_test[df_test['gender'] == 'F']
        print('Males: ', m_data.shape, 'Females: ', f_data.shape)
        df_test = df_test.drop(['gender'], axis=1)
        y_test = np.array([0 if tmp_y == 'M' else 1 for tmp_y in tags])

        print('All shape: ', df_train.shape, y_train.shape, df_test.shape, y_test.shape)

        return df_train, y_train, df_test, y_test



    else:
        if predict:
            tags = df_data.id
        else:
            tags = df_data.gender
            m_data = df_data[df_data['gender'] == 'M']
            f_data = df_data[df_data['gender'] == 'F']
            print('Males: ', m_data.shape, 'Females: ', f_data.shape)
        df_data = df_data.drop(['gender'], axis=1)
        if not predict:
            y = np.array([0 if tmp_y=='M' else 1 for tmp_y in tags])
        else:
            y = np.array([tmp_y for tmp_y in tags])
        return df_data, y


if __name__ == '__main__':
    start_time = time.time()
    # run from command line
    # e.g. python3 gender_classification.py --input './pan17-author-profiling-training-dataset-2017-03-10' --output results --language en
    argparser = argparse.ArgumentParser(description='Clin gender evaluation')
    argparser.add_argument('-c', '--input', dest='input', type=str,
                           default='data/weebit',
                           help='Choose input trainset')
    # args = argparser.parse_args()
    main()

    print("--- Model creation in minutes ---", round(((time.time() - start_time) / 60), 2))
    print("--- Training & Testing in minutes ---", round(((time.time() - start_time) / 60), 2))





