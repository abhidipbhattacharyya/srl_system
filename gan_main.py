import sys
import os
import re
import time
import numpy as np
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
from model.cgan import *
from os.path import isfile
from util.utils import *

if __name__ == '__main__':

    fname = EMBEDDING
    load_kwargs={"vocab_size": 400000, "dim": 300}
    w = Embedding.from_glove(fname, **load_kwargs)
    dL = DataLoader()
    wdic = WordDictionary(w)
    tdic = TagDictionary()
    dL.readSRLData(sys.argv[1], wdic, tdic, False)
    batched_data = Process.create_batch_data(dL.sentences,cfg.BATCH_SIZE, wdic, tdic)

    model = lstm_crf(len(wdic.word2idx), len(tdic.tag2idx), False)
    epoch = load_checkpoint('trained_model/model.epoch20', model)

    generator = Generator(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight())
    generator.set_crf(model.crf)
    discriminator = Discriminator(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight())

    epoch = 0
    print("training model...")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G)#, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D)#, betas=(opt.b1, opt.b2))
    criterion = myLoss()
    for ei in range(epoch+1, cfg.NUM_EPCH+epoch+1):
        loss_sum = 0
        for x, f, y in batched_data:
            noise = model.decode(x,f)
            noise = [ns[1:] for ns in noise]# the first one is SOS
            noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.

            #print('shape of noise {} {}'.format(noise.shape, x.shape))
            #print(f)
            result = generator(x,f,noise,y)
            result = [rs[1:] for rs in result]
            result = Process.process_noise(result)#pad it again, make tensor.
            #print(result)
            y = y[:,1:]
            #print('result shape:{};  actual shape:{}'.format(result.shape, y.shape))
            validity_gen = discriminator(x,f,result)# act sen, syn tag
            validity_act = discriminator(x,f,y)# act sen, act tag

            ##-- find the hardest negative----##
            validity_wrong = discriminator(x,f,y[0].expand_as(y)).squeeze(-1)
            #print('worng score:{}'.format(validity_wrong))
            for i in range(1,y.shape[0]):
                vw = discriminator(x,f,y[i].expand_as(y)).squeeze(-1)
                #print(vw)
                validity_wrong = torch.cat((validity_wrong, vw),0) #act tag, WRONG sen

            validity_wrong = validity_wrong.view(y.shape[0],-1)

            #print('val score:{}'.format(validity_gen))
            #print('act score:{}'.format(validity_act))
            #print('worng score:{}'.format(validity_wrong))

            G_loss, D_loss = criterion(validity_act, validity_gen, validity_wrong.t())
            print('loss=={}, {}'.format(G_loss, D_loss))

            optimizer_D.zero_grad()
            D_loss.backward(retain_graph=True)      # reusing computational graph
            optimizer_D.step()
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()
