import numpy as np
import torch
from torch.autograd import Variable
import pickle
from collections import Counter
from torch import nn
import torch.nn.functional as F
from nltk.tag import PerceptronTagger
from nltk.corpus import alpino as alp
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import PunktSentenceTokenizer
training_corpus = list(alp.tagged_sents())
tagger = PerceptronTagger(load=True)
tagger.train(training_corpus)
wordTokenizer = WordPunctTokenizer()
sentTokenizer = PunktSentenceTokenizer()


def generate_vocabulary(data, vocabulary_size):
    all_data = " ".join(data)
    print(all_data[:100])
    words = [word for sent in sentTokenizer.tokenize(all_data) for word in wordTokenizer.tokenize(sent)]
    counter = Counter(words)

    # most_common() produces k frequently encountered
    # input values and their respective counts.
    most_common = counter.most_common(vocabulary_size)
    vocabulary = set([word for word, count in most_common])
    return vocabulary


class Dictionary(object):
    def __init__(self, words=True):
        if words:
            #unknown is one, zero is reserved for padding
            self.word2idx = {'<unk>':1}
            self.idx2word = ['<unk>']
        else:
            self.word2idx = {}
            self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, vocabulary, train_x, train_y, test_x, test_y, batch_size, max_length=False, words=True, vocab_only=False, pos_tag=False):
        self.dictionary = Dictionary(words)
        self.vocabulary = vocabulary
        self.batch_size = batch_size
        self.max_length = max_length
        self.words = words
        self.pos_tag = pos_tag
        self.vocab_only = vocab_only
        if not pos_tag:
            self.train_x, self.train_y = self.tokenize(train_x, train_y, words, vocab_only)
            self.test_x, self.test_y = self.tokenize(test_x, test_y, words, vocab_only, dict_exist=True)
        else:

            self.tagger = tagger
            self.train_x, self.train_x_pos, self.train_y = self.tokenize(train_x, train_y, words, vocab_only, pos_tag=True)
            self.test_x, self.test_x_pos, self.test_y = self.tokenize(test_x, test_y, words, vocab_only, dict_exist=True, pos_tag=True)



    def tokenize(self, data_x, data_y, words, vocab_only, dict_exist=False, pos_tag=False):

        # Add words to the dictionary
        all_data = []
        for i, text in enumerate(data_x):
            if words:
                tokens = [word for sent in sentTokenizer.tokenize(text) for word in wordTokenizer.tokenize(sent)]
                if pos_tag:
                    pos_tokens = sentTokenizer.tokenize(text)
                    # use average perceptron tagger
                    pos_tags = [self.tagger.tag(wordTokenizer.tokenize(sent)) for sent in pos_tokens]
                    pos_tags = [tag for sent in pos_tags for word, tag in sent]
                    #print(pos_tags)
            else:
                tokens = [c for c in text]
            if not dict_exist:
                for token in tokens:
                    if words:
                        if token in self.vocabulary:
                            self.dictionary.add_word(token)
                    else:
                        self.dictionary.add_word(token)
                if pos_tag:
                    for token in pos_tags:
                        self.dictionary.add_word(token)

            if vocab_only:
                tokens = [token for token in tokens if token in self.vocabulary]
            if not dict_exist and self.max_length < len(tokens):
                self.max_length = len(tokens)
            if not pos_tag:
                all_data.append((len(tokens), tokens, data_y[i]))
            else:
                all_data.append((len(tokens), tokens, pos_tags, data_y[i]))
        print(len(self.dictionary.idx2word), self.dictionary.idx2word)
        all_data = sorted(all_data, key=lambda x: x[0])
        cut_idx = len(all_data) - (len(all_data) % self.batch_size)
        all_data = all_data[:cut_idx]
        if not pos_tag:
            data_x = [x[1] for x in all_data]
            data_y = [x[2] for x in all_data]
        if pos_tag:
            data_x = [x[1] for x in all_data]
            data_pos = [x[2] for x in all_data]
            data_y = [x[3] for x in all_data]
        batchified_data_x = []
        batchified_data_y = []
        if pos_tag:
            batchified_data_pos = []
        for i in range(0, len(all_data), self.batch_size):
            batchified_data_x.append(data_x[i: i + self.batch_size])
            batchified_data_y.append(torch.as_tensor(np.array(data_y[i: i + self.batch_size])))
            if pos_tag:
                batchified_data_pos.append(data_pos[i: i + self.batch_size])

        normalized_batchified_data_x = []
        for batch in batchified_data_x:
            if self.max_length:
                max_length = min(len(batch[-1]), self.max_length)
            else:
                max_length = len(batch[-1])
            np_data = np.zeros((self.batch_size, max_length), dtype='int32')
            for i, tokens in enumerate(batch):
                for j in range(max_length):
                    try:
                       token = tokens[j]
                    except:
                        continue
                    if words:
                        if token in self.vocabulary and token in self.dictionary.word2idx:
                            np_data[i,j] = self.dictionary.word2idx[token]
                        else:
                            np_data[i,j] = self.dictionary.word2idx['<unk>']
                    else:
                        np_data[i, j] = self.dictionary.word2idx[token]
                #print(tokens)
                #print(np_data[i].tolist())
                #print([self.dictionary.idx2word[x - 1] for x in np_data[i].tolist()])
            normalized_batchified_data_x.append(torch.LongTensor(np_data))
            #print('Shape of data tensor:', np_data.shape, np_data)

            if pos_tag:
                normalized_batchified_data_pos = []
                for batch in batchified_data_pos:
                    if self.max_length:
                        max_length = min(len(batch[-1]), self.max_length)
                    else:
                        max_length = len(batch[-1])
                    np_data = np.zeros((self.batch_size, max_length), dtype='int32')
                    for i, tokens in enumerate(batch):
                        for j in range(max_length):
                            try:
                                token = tokens[j]
                            except:
                                continue
                            if token in self.dictionary.word2idx:
                                np_data[i, j] = self.dictionary.word2idx[token]
                            else:
                                np_data[i, j] = self.dictionary.word2idx['<unk>']

                    normalized_batchified_data_pos.append(torch.LongTensor(np_data))
        if pos_tag:
            return normalized_batchified_data_x, normalized_batchified_data_pos, batchified_data_y
        return normalized_batchified_data_x, batchified_data_y


    def generate_embeddings_matrix(self, embeddings, embedding_dim):

        embedding_matrix = np.zeros((len(self.dictionary.word2idx) + 3, embedding_dim))
        for token, i in self.dictionary.word2idx.items():
            try:
                embedding_vector = embeddings.get_vector(token)
                embedding_matrix[i + 2] = embedding_vector
            except:
                embedding_matrix[i + 2] = np.random.normal(scale=0.6, size=(embedding_dim,))
        return torch.FloatTensor(embedding_matrix)



def batchify(data_x, data_y, batch_size):
    doc_length = data_x.size(-1)
    #print(doc_length)

    nbatch = data_x.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data_x = data_x.narrow(0, 0, nbatch * batch_size)
    data_y = data_y.narrow(0, 0, nbatch * batch_size)

    # Evenly divide the data across the batch_size batches.
    data_x = data_x.view(-1, batch_size, doc_length)
    data_x = data_x

    data_y = data_y.view(-1, batch_size)
    #print(data_y)
    return data_x, data_y


def get_batch(source_x, source_y, i):
    data = Variable(source_x[i]).cuda()
    target = Variable(source_y[i]).cuda()
    #target = target.unsqueeze(1)
    return data, target


def create_emb_layer(weights_matrix, non_trainable=True):
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim




class NetLstm(nn.Module):
    def __init__(self, weights_matrix, vocab_size, label_size, batch_size, hidden_dim=128, use_gpu=True):
        super(NetLstm, self).__init__()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.hidden_dim = hidden_dim

        #self.word_embeddings,num_embeddings, embedding_dim = create_emb_layer(weights_matrix, non_trainable=False)

        self.word_embeddings = nn.Embedding(vocab_size, 200, padding_idx=0)
        self.lstm = nn.LSTM(200, self.hidden_dim, dropout=0.1, bidirectional=True)
        self.hidden2label = nn.Linear(8 * self.hidden_dim, label_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.hidden = self.init_hidden()


        self.pos_embeddings = nn.Embedding(vocab_size, 200, padding_idx=0)
        self.pos_lstm = nn.LSTM(200, self.hidden_dim, dropout=0.1, bidirectional=True)




    def init_hidden(self):
        if self.use_gpu:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(2, self.batch_size, self.hidden_dim))
        return (h0, c0)


    def embedded_dropout(self, embeds, words, dropout=0.1):
        emb = embeds(words)
        mask = embeds.weight.data.new_empty((embeds.weight.size(0), 1)).bernoulli_(1 - dropout) / (1 - dropout)
        m = torch.gather(mask.expand(embeds.weight.size(0), words.size(1)), 0, words)
        return (emb * m.unsqueeze(2).expand_as(emb)).squeeze()


    def forward(self, text, pos):
        #print("Text shape: ", text.size())
        embeds = self.word_embeddings(text)

        #print("Embeds size: ", embeds.size())
        embeds = embeds.view(text.size(0), self.batch_size, -1)
        #print("Embeds size 2: ", embeds.size())
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        #print("Lstm size: ", lstm_out.size())
        #for seq in lstm_out:
        #    print(seq.size())
        #return 0
        # print(relu_out.size(), relu_out)
        #avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        #print('Adaptive avg pooling', avg_pool.size())

        max_pool = F.adaptive_max_pool1d(lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        #print('Max pooling', max_pool.size())

        pos_embeds = self.pos_embeddings(pos)
        # print("Embeds size: ", embeds.size())
        pos_embeds = pos_embeds.view(pos.size(0), self.batch_size, -1)
        # print("Embeds size 2: ", embeds.size())
        self.hidden = self.init_hidden()
        pos_lstm_out, self.hidden = self.pos_lstm(pos_embeds, self.hidden)
        pos_avg_pool = F.adaptive_avg_pool1d(pos_lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)
        pos_max_pool = F.adaptive_max_pool1d(pos_lstm_out.permute(1, 2, 0), 1).view(self.batch_size, -1)

        outp = torch.cat([max_pool, avg_pool, pos_max_pool, pos_avg_pool], dim=1)
        y = self.dropout(self.relu(outp))
        #print('Y', y.size())
        y = self.hidden2label(y)


        return y
