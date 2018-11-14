import sys
import os
import re
import time
from reader.DataLoader import DataLoader
from reader.Dictionary import *
from embedding.embedding import Embedding
from feature_extraction.process import Process
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import configuration.config as cfg
from model.base_model_BLSTM import *
from os.path import isfile
from util.utils import *
from util.conll_utils import *
from model.cgan import *
from evaluation import *

PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3

def run_model(model, idx_to_tag, data):
    batch = []
    z = len(data)
    while len(data) < BATCH_SIZE:
        data.append([-1, "", [EOS_IDX]])
    data.sort(key = lambda x: len(x[2]), reverse = True)
    batch_len = len(data[0][2])
    batch = [x + [PAD_IDX] * (batch_len - len(x)) for _, _, x in data]
    result = model.decode(LongTensor(batch))
    for i in range(z):
        data[i] = data[i][:-1] + [idx_to_tag[j] for j in result[i]]
    return [(x[1], x[2]) for x in sorted(data[:z])]

if __name__ == '__main__':
    #--load things---##
    print('usage: python predict.py devfile goldfile mdoel')
    wdic = load_word_to_idx(cfg.WORD_2_IDX_PATH)
    tdic = load_tag_to_idx(cfg.TAG_2_IDX_PATH)

    model = lstm_crf(len(wdic.word2idx), len(tdic.tag2idx), False)
    epoch = load_checkpoint(CRF_READ_GAN, model)

    generator = Generator(len(wdic.word2idx), len(tdic.tag2idx), False)
    model.eval()

    epoch = load_checkpoint(sys.argv[3], generator)
    print(generator.crf.parameters())
    #print(tdic.idx2tag)
    dL = DataLoader()
    dL.readSRLTestData(sys.argv[1], wdic, tdic, True)
    #print(dL.sentences[0])
    batched_data = Process.create_batch_data_dev(dL.sentences, cfg.BATCH_SIZE, wdic, tdic) #create_batch_data_testing(dL.sentences, cfg.BATCH_SIZE, wdic, tdic)
    #print(batch_data)
    all_result = [[]]*len(dL.sentences)
    for x,f,y, indices in batched_data:
        noise = model.decode(x,f)
        noise = [ns[1:] for ns in noise]# the first one is SOS
        noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.

        #print('shape of noise {} {}'.format(noise.shape, x.shape))
        #print(f)
        result = generator(x,f,noise,y)
        result = [[tdic.getTag(j) for j in result[i]] for i in range(len(result))]
        for i, idx in enumerate(indices):
            slen = len(result[i])
            all_result[idx] = result[i][1:slen-1] if USE_SE else result[i][0:slen-1]
            #all_result[idx] = bio_to_se( all_result[idx])

    #print(all_result)
    evaluate(all_result, sys.arg[2])
    #print_to_conll(all_result, 'data/testGold.txt', 'data/testPred.txt')
    print('done')
