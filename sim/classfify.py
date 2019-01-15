import os
rootpath = os.path.dirname(__file__)
import sys
sys.path.append(rootpath)
from keras.models import Model
from keras.layers import *
from keras.constraints import unit_norm
from softmax import *
from keras.callbacks import Callback
from keras.models import load_model
from keras import backend as K
import numpy as np
import pdb


class ClassifyModel():
    def __init__(self, params, modelfile, vocabfile, stopfile):
        #print(modelfile)
        self.__nclass = params.nclass
        self.__maxlen = params.maxlen
        self.__wordsize = params.word_size
        self.__cuda = params.cuda
        self.__vocabsize = 0
        self.__char2id = dict()
        self.__id2char = dict()
        self.__stopwords = list()
        self.__queslist = list()
        self.__knowledgelist = list()
        self.__veclist = None
        self.__model = None
        self.__encoder = None
        self.load_stopwords(stopfile)
        self.load_vocab(vocabfile)
        self.__net(modelfile)


    def load_stopwords(self, path):
        with open(path, "r") as f:
            for line in f.readlines():
                text = line.strip("\r\n")
                self.__stopwords.append(text)

    def load_vocab(self, path):
        with open(path, "r") as f:
            for line in f.readlines():
                text = line.strip("\r\n")
                gs = text.split("\001")
                self.__char2id[gs[0]] = 2 + int(gs[1])
                self.__id2char[2 + int(gs[1])] = gs[0]
        self.__vocabsize = len(self.__char2id)

    def __string2id(self,s):
        _ = [self.__char2id.get(i, 1) for i in s[:self.__maxlen]]
        _ = _ + [0] * (self.__maxlen - len(_))
        return _

    def __net(self, modelpath):
            x_in = Input(shape=(self.__maxlen,))
            x_embedded = Embedding(self.__vocabsize+2,self.__wordsize)(x_in)
            x_embedded = BatchNormalization()(x_embedded)
            x_embedded = Activation('relu', )(x_embedded)
            reshape = Reshape((self.__maxlen, self.__wordsize, 1))(x_embedded)
            self.num_filters = 512
            self.filter_sizes = [3, 4, 5]
            conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.__wordsize), padding='valid',
                            kernel_initializer='normal', activation='relu')(reshape)
            conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.__wordsize), padding='valid',
                            kernel_initializer='normal', activation='relu')(reshape)
            conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.__wordsize), padding='valid',
                            kernel_initializer='normal', activation='relu')(reshape)

            maxpool_0 = MaxPool2D(pool_size=(self.__maxlen - self.filter_sizes[0] + 1, 1), strides=(1, 1),
                                  padding='valid')(
                conv_0)
            maxpool_1 = MaxPool2D(pool_size=(self.__maxlen - self.filter_sizes[1] + 1, 1), strides=(1, 1),
                                  padding='valid')(
                conv_1)
            maxpool_2 = MaxPool2D(pool_size=(self.__maxlen - self.filter_sizes[2] + 1, 1), strides=(1, 1),
                                  padding='valid')(
                conv_2)
            concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
            flatten = Flatten()(concatenated_tensor)
            reshape1 = Reshape((3, self.num_filters))(flatten)
            reshape1 = BatchNormalization()(reshape1)
            #reshape1 = Activation('relu', )(reshape1)
            if self.__cuda:
                x = CuDNNGRU(self.__wordsize)(reshape1)
            else:
                x = Bidirectional(GRU(self.__wordsize,return_sequences=True))(reshape1)
            #print('x',x)
            #x = BatchNormalization()(x)
            timestep = TimeDistributed(Dense(1))(x)
            flatterns = Flatten()(timestep)
            flatterns = BatchNormalization()(flatterns)
            attention_weight = Activation('softmax')(flatterns)
            attention_weight = RepeatVector(2 * self.__wordsize)(attention_weight)
            attention_weight = Permute([2, 1])(attention_weight)
            sent_representation = multiply([x, attention_weight])
            sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
            x = sent_representation
            x = Lambda(lambda x: K.l2_normalize(x, 1))(x)
            pred = Dense(self.__nclass, use_bias=False,kernel_constraint=unit_norm())(x)
            self.__encoder = Model(x_in, x)
            self.__model = Model(x_in, pred)
            self.__model.compile(loss=sparse_amsoftmax_loss,optimizer='adam',metrics=['sparse_categorical_accuracy'])
            self.__model.load_weights(modelpath)

    def __filter(self, sen):
        train_data = ""
        for ch in sen:
            if ch in self.__stopwords:
                continue
            train_data += ch
        return train_data

    def get_sentence_vector(self, questr):
        questr1 = questr
        if isinstance(questr, list) == True:
            questr1 = "".join(questr1)
        string = self.__filter(questr1)
        vec = self.__encoder.predict(np.array([self.__string2id(string)]))[0]
        return vec,None

pass
