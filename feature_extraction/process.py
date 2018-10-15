import numpy as np
import os
import sys
import torch
#from torch import LongTensor
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from configuration.config import *
#from ..reader.Dictionary import Dictionary
##TODO - should we move the indexing in reading?
##TODO- batching data from https://github.com/threelittlemonkeys/lstm-crf-pytorch/blob/master/sequence-labelling/prepare.py
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3
CUDA = torch.cuda.is_available()
def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x
class Process:
    @staticmethod
    def tenorise(sentence, wDic, tDic):
        sen = [wDic.getIndex(w) for w in sentence[1]]
        sen = torch.tensor(sen, dtype=torch.long)
        tag = [tDic.getIndex(t) for t in sentence[2]]
        tag = torch.tensor(tag, dtype=torch.long)
        return (sentence[0],sen,tag)

    @staticmethod
    def create_batch_data(sentences, batchsize, wDic, tDic):
        num_of_batch = len(sentences) // batchsize
        batch_x = []
        batch_f =[]
        batch_y = []

        data = []
        feature_batches = []
        index =0
        max_train_length = len(sentences[0][1])# the first element has max seq length
        batch_len = MAX_SEQ_LEN#max_train_length#0 # maximum sequence length of a mini-batch

        for sen in sentences:
            index = index+1
            pred = int(sen[0])
            words = sen[1]
            tags = sen[2]
            seq_len = len(sen[1])
            if len(batch_x)==0:# the first line has the maximum sequence length
                batch_len = seq_len

            pad = [PAD] * (batch_len - seq_len)
            if USE_SE:
                padded_words = [SOS]+words[:seq_len] + [EOS] + pad
                pred = pred +1
            else:
                padded_words = words[:seq_len] + [EOS] + pad
            padded_tags = [SOS] + tags[:seq_len] + [EOS] + pad

            padded_pred = [0]*len(padded_words)
            padded_pred[pred] = 1

            padded_words = [wDic.getIndex(w) for w in padded_words]
            padded_tags = [tDic.getIndex(t) for t in padded_tags]

            batch_x.append(padded_words)
            batch_y.append(padded_tags)
            batch_f.append([padded_pred])
            #print('{} vs{}'.format(index, len(sentences) ))
            ## # TODO: why no SOS marking???
            if len(batch_x) == batchsize or index == len(sentences): ##TODO for the last batch.
                data.append((LongTensor(batch_x), LongTensor(batch_f), LongTensor(batch_y))) # append a mini-batch
                batch_x = []
                batch_y = []
                batch_f = []
        return data

    @staticmethod
    def create_batch_data_dev(sentences, batchsize, wDic, tDic):
        num_of_batch = len(sentences) // batchsize
        batch_x = []
        batch_f =[]
        batch_y = []
        batch_ac_indx =[] # keep track of the actual indices of sentences
        batch_len = 0 # maximum sequence length of a mini-batch
        data = []
        feature_batches = []
        index =0
        #max_dev_length = max([len(s[1]) for s in sentences])
        for sen in sentences:
            index = index+1
            pred = int(sen[0])
            words = sen[1]
            tags = sen[2]
            sen_ac_indx = sen[3]
            seq_len = len(sen[1])
            if len(batch_x)==0:# the first line has the maximum sequence length
                batch_len = seq_len

            pad = [PAD] * (batch_len - seq_len)
            if USE_SE:
                padded_words = [SOS]+words[:seq_len] + [EOS] + pad
                pred = pred +1
            else:
                padded_words = words[:seq_len] + [EOS] + pad
            padded_tags = [SOS] + tags[:seq_len] + [EOS] + pad

            padded_pred = [0]*len(padded_words)
            padded_pred[pred] = 1

            padded_words = [wDic.getIndex(w) for w in padded_words]
            padded_tags = [tDic.getIndex(t) for t in padded_tags]

            batch_x.append(padded_words)
            batch_y.append(padded_tags)
            batch_f.append([padded_pred])
            batch_ac_indx.append(sen_ac_indx)
            #print('{} vs{}'.format(index, len(sentences) ))
            ## # TODO: why no SOS marking???
            if len(batch_x) == batchsize or index == len(sentences): ##TODO for the last batch.
                data.append((LongTensor(batch_x), LongTensor(batch_f), LongTensor(batch_y), batch_ac_indx )) # append a mini-batch
                batch_x = []
                batch_y = []
                batch_f = []
                batch_ac_indx =[]
        return data

    @staticmethod
    def create_batch_data_testing(sentences, batchsize, wDic, tDic):
        num_of_batch = len(sentences) // batchsize
        batch_x = []
        batch_f =[]
        #batch_y = []
        batch_len = 0 # maximum sequence length of a mini-batch
        data = []
        index =0
        for sen in sentences:
            index = index+1
            pred = int(sen[0])
            words = sen[1]
            #tags = sen[2]
            seq_len = len(sen[1])
            if len(batch_x)==0:# the first line has the maximum sequence length
                batch_len = seq_len

            pad = [PAD] * (batch_len - seq_len)
            if USE_SE:
                padded_words = [SOS]+words[:seq_len] + [EOS] + pad
                pred = pred +1
            else:
                padded_words = words[:seq_len] + [EOS] + pad

            #padded_tags = [SOS] + tags[:seq_len] + [EOS] + pad

            padded_words = [wDic.getIndex(w) for w in padded_words]
            #padded_tags = [tDic.getIndex(t) for t in padded_tags]
            padded_pred = [0]*len(padded_words)
            padded_pred[pred] = 1 #if there is a sos in sentence pred =pred+1

            batch_x.append(padded_words)
            batch_f.append([padded_pred])
            #batch_y.append(padded_tags)
            #print('{} vs{}'.format(index, len(sentences) ))
            ## # TODO: why no SOS marking???
            if len(batch_x) == batchsize or index == len(sentences): ##TODO for the last batch.
                data.append((LongTensor(batch_x),LongTensor(batch_f))) # append a mini-batch
                batch_x = []
                batch_f =[]
                #batch_y = []
        return data
    @staticmethod
    def process_noise(noises): # already in batch with EOS (and may be SOS)
        batch_len = len(noises[0])
        batch_noise = []
        for ns in noises:
            #print(ns)
            seq_len = len(ns)

            pad = [PAD_IDX] * (batch_len - seq_len)
            ns = ns +[EOS_IDX]+ pad
            batch_noise.append(ns)
        return LongTensor(batch_noise)
