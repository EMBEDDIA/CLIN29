# -*- coding: utf-8 -*-

import pandas as pd
import re
import gensim
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from architecture import generate_vocabulary, Corpus, batchify, get_batch, NetLstm
import string
from collections import defaultdict

from torch import nn
import pickle
import torch
import os
from collections import defaultdict


# Set the random seed manually for reproducibility.
torch.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(123)
if torch.cuda.is_available():
    print(torch.cuda.is_available())



def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


def preprocess(df_data):
    #df_data['text'] = df_data['text'].map(lambda x: fix_sentences(x))
    df_data['text_clean_r'] = df_data['text'].map(lambda x: remove_hashtags(x, '#HASHTAG'))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_url(x, "HTTPURL"))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_mentions(x, '@MENTION'))
    df_data['text_clean'] = df_data['text'].map(lambda x: remove_hashtags(x, ''))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_url(x, ""))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_mentions(x, ''))
    df_data['text_clean'] = df_data['text_clean_r']
    df_data = df_data.drop('text_clean_r', 1)

    return df_data


def train_and_test(xtrain, ytrain, xval, yval, num_epoch, test_set_name):
    batch_size = 16
    EMBEDDING_DIM = 300
    vocabulary_size = 30000

    '''if os.path.exists('models/w2v_model.pk'):
        w2v = pickle.load(open('models/w2v_model.pk', 'rb'))
    else:
        w2v = gensim.models.KeyedVectors.load_word2vec_format(os.path.join('embeddings', 'wiki.nl.vec'))
        with open('models/w2v_model.pk', 'wb') as f:
            pickle.dump(w2v, f)

    print('pretrained embeddings loaded')'''

    print("Fold shape: ", xtrain.shape, xval.shape)

    #tfidf_unigram_embeddings = TfidfVectorizer(ngram_range=(1, 1), min_df=4, sublinear_tf=True)
    #tfidf_unigram_embeddings = tfidf_unigram_embeddings.fit(xtrain.text_clean)


    vocabulary = generate_vocabulary(xtrain.text_clean.tolist(), vocabulary_size)
    #vocabulary = tfidf_unigram_embeddings.vocabulary_
    #print(vocabulary)
    corpus = Corpus(vocabulary, xtrain.text_clean.tolist(), ytrain, xval.text_clean.tolist(), yval, batch_size, pos_tag=True)

    #weights_matrix = corpus.generate_embeddings_matrix(w2v, EMBEDDING_DIM)
    weights_matrix = []
    #print("Weights matrix: ", weights_matrix)

    num_classes = len(set(list(ytrain)))
    print("Num classes: ", num_classes)

    print("corpus generated")
    #train_data_x, train_data_y = batchify(corpus.train_x, corpus.train_y, batch_size)
    #test_data_x, test_data_y = batchify(corpus.test_x, corpus.test_y, batch_size)
    vocab_size = len(corpus.dictionary) + 1
    print("vocabulary_size: ", vocab_size)
    print('batchification done')

    with open('models/' + test_set_name + '_corpus.pk', 'wb') as f:
        print("Saving corpus")
        pickle.dump(corpus, f)

    model, best_accuracy = train(corpus.train_x, corpus.train_x_pos, corpus.train_y, corpus.test_x, corpus.test_x_pos, corpus.test_y, weights_matrix, vocab_size, batch_size, num_classes, num_epoch, test_set_name)

    return model, corpus




def train(data_x, data_pos, data_y, test_x, test_pos, test_y, weights_matrix, vocab_size, batch_size, num_classes, num_epoch, test_set_name):
    print('Starting training')

    model = NetLstm(weights_matrix, vocab_size, num_classes, batch_size)
    model.cuda()


    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    best_accuracy = 0

    # Step 3. Run our forward pass.
    total_pred = []
    total_true = []
    for epoch in range(num_epoch):
        print()
        print("Epoch: ", epoch + 1)
        print()

        for i in range(len(data_x)):
            batch_x, batch_y = get_batch(data_x, data_y, i)
            batch_pos, batch_y = get_batch(data_pos, data_y, i)
            optimizer.zero_grad()
            model.hidden = model.init_hidden()
            predicted_batch = model(batch_x.t(), batch_pos.t())
            #print(predicted_y)
            #print(batch_y)
            maxes = []
            #print(predicted_batch)
            for prediction in predicted_batch:
                values, idx = prediction.max(0)
                maxes.append(idx.item())

            true_y = list(batch_y.cpu().numpy())
            #print(accuracy_score(maxes, true_y), maxes, true_y)


            loss = loss_function(predicted_batch, batch_y)
            #print("Loss: ", loss)
            loss.backward()
            optimizer.step()
            total_pred.extend(maxes)
            total_true.extend(true_y)
        accuracy = accuracy_score(total_pred, total_true)
        print("Train accuracy: ", accuracy)

        #testing
        print("Testing")
        total_pred = []
        total_true = []
        for i in range(len(test_x)):
            batch_x, batch_y = get_batch(test_x, test_y, i)
            batch_pos, batch_y = get_batch(test_pos, test_y, i)
            #print(batch_x.size(), batch_x)
            #print(batch_y)

            model.hidden = model.init_hidden()
            predicted_batch = model(batch_x.t(), batch_pos.t())
            #print(predicted_y)
            #print(batch_y)
            maxes = []
            #print(predicted_batch)
            for prediction in predicted_batch:
                values, idx = prediction.max(0)
                maxes.append(idx.item())

            true_y = list(batch_y.cpu().numpy())

            #print(predicted_y.size(), batch_y.size())
            #print("Loss: ", loss)

            # calc testing acc
            #print("Test accuracy: ", accuracy_score(maxes, true_y), maxes, true_y)

            total_pred.extend(maxes)
            total_true.extend(true_y)
        accuracy = accuracy_score(total_pred, total_true)
        if accuracy > best_accuracy:
            with open("models/" + test_set_name + "_model.pt", 'wb') as f:
                print('Saving model')
                torch.save(model, f)
            best_accuracy = accuracy
        print("Test accuracy: ", accuracy, "Best accuracy: ", best_accuracy)
    return model, best_accuracy


def test(xval, yval, model, corpus, filename, test=False):
    vocab_size = len(corpus.dictionary) + 1
    print("vocabulary_size: ", vocab_size)

    corpus.test_x, corpus.test_x_pos, corpus.test_y = corpus.tokenize(xval.text_clean.tolist(), yval, corpus.words, corpus.vocab_only, dict_exist=True, pos_tag=True)
    weights_matrix = []
    #print("Weights matrix: ", weights_matrix)

    predict(model, corpus.test_x, corpus.test_y, corpus.test_x_pos, filename, test=test)

    print("X: ", len(corpus.test_x))
    print("Y: ", len(corpus.test_y))
    print("POS: ", len(corpus.test_x_pos))


def predict(model, test_x, test_y, test_pos, filename, test=False):
    model.cuda()


    # testing
    print("Testing")
    total_pred = []
    total_true = []
    for i in range(len(test_x)):
        batch_x, batch_y = get_batch(test_x, test_y, i)
        batch_pos, batch_y = get_batch(test_pos, test_y, i)
        # print(batch_x.size(), batch_x)
        # print(batch_y)

        model.hidden = model.init_hidden()
        predicted_batch = model(batch_x.t(), batch_pos.t())
        # print(predicted_y)
        # print(batch_y)
        maxes = []
        # print(predicted_batch)
        for prediction in predicted_batch:
            values, idx = prediction.max(0)
            maxes.append(idx.item())
        total_pred.extend(maxes)

        true_y = list(batch_y.cpu().numpy())
        total_true.extend(true_y)

        # print(predicted_y.size(), batch_y.size())
        # print("Loss: ", loss)

        # calc testing acc
        # print("Test accuracy: ", accuracy_score(maxes, true_y), maxes, true_y)
    if test:
        accuracy = accuracy_score(total_pred, total_true)
        print("Test accuracy: ", accuracy, "Best accuracy: ", accuracy)
    else:
        path = 'predictions/' + filename
        if os.path.exists(path):
            os.remove(path)
        total_pred = ['M' if tmp_y==0 else 'F' for tmp_y in total_pred]
        pred_dict = defaultdict(list)
        for i in range(len(total_true)):
            pred_dict[str(total_true[i])].append(str(total_pred[i]))
        if filename.endswith('youtube_1'):
            gold = "data/gold/GxG_Youtube_gold.txt"
        if filename.endswith('news_1'):
            gold = "data/gold/GxG_News_gold.txt"
        if filename.endswith('twitter_1'):
            gold = "data/gold/GxG_twitter_gold.txt"
        with open(gold, 'r', encoding='utf8') as f:
            for line in f:
                id, pred = line.split()
                pred_dict[id].append(pred)
        all = 0
        correct = 0
        for results in pred_dict.values():
            all += 1
            if results[0] == results[1]:
                correct += 1
        print("Accuracy on gold: ", correct/all)


        with open(path, 'a', encoding='utf8') as f:
            for i in range(len(total_true)):
                f.write(str(total_true[i]) + ' ' + str(total_pred[i]) + '\n')












