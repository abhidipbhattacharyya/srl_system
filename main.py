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
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3

if __name__ == '__main__':

    fname = EMBEDDING
    load_kwargs={"vocab_size": 400000, "dim": 300}
    w = Embedding.from_glove(fname, **load_kwargs)
    print('embedding read')
    #START_TAG = "<START>"
    #STOP_TAG = "<STOP>"
    #w.vocabulary.word_id
    dL = DataLoader()
    wdic = WordDictionary(w)
    tdic = TagDictionary()
    dL.readSRLData(sys.argv[1], wdic, tdic, False)
    print('data read from {}'.format(sys.argv[1]))

    print('saving dictionaries')
    save_word_to_idx(cfg.WORD_2_IDX_PATH,wdic)
    save_tag_to_idx(cfg.TAG_2_IDX_PATH,tdic)

    print('processing data for batching')
    batched_data = Process.create_batch_data(dL.sentences,cfg.BATCH_SIZE, wdic, tdic)

    #print(dL.sentences[0])
    #print(tdic.tag2idx)
    #print(tdic.idx2tag)
    #print('weight type {}'.format(wdic.getWeight().type))

    #sen = Feature.tenorise(dL.sentences[0], wdic, tdic)
    #print(sen)
    #fname = os.path.join(path,'model_arg_TD_')
    filename = os.path.join(cfg.TRAINDED_MODEL_PATH,sys.argv[2])
    model = lstm_crf(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight()) #cfg.EMBED_SIZE, cfg.HIDDEN_SIZE, cfg.NUM_DIRS, cfg.NUM_LAYERS, cfg.BIDIRECTIONAL, cfg.DROPOUT,
    optim = torch.optim.SGD(model.parameters(), lr = cfg.LEARNING_RATE, weight_decay = cfg.WEIGHT_DECAY)
    epoch = 0#load_checkpoint(filename, model) if isfile(sys.argv[1]) else 0
    filename = re.sub("\.epoch[0-9]+$", "", filename)

    print(model)
    print("training model...")
    for ei in range(epoch+1, cfg.NUM_EPCH+epoch+1):
        loss_sum = 0
        timer = time.time()
        for x, f, y in batched_data:
            #print('shape of batch{} '.format(x.shape))
            model.zero_grad()
            loss = torch.mean(model(x,f, y))
            loss.backward() # compute gradients
            optim.step() # update parameters
            loss = scalar(loss)
            loss_sum += loss
        timer = time.time() - timer
        loss_sum /= len(batched_data)
        if ei % cfg.SAVE_EVERY == 0 or ei == epoch + cfg.NUM_EPCH:
            save_checkpoint(filename, model, ei, loss_sum, timer)


    print('done')#
