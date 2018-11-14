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
BCE_loss = nn.BCELoss()

def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    fname = EMBEDDING
    filename = os.path.join(cfg.TRAINDED_MODEL_PATH,sys.argv[2])
    load_kwargs={"vocab_size": 400000, "dim": 300}
    w = Embedding.from_glove(fname, **load_kwargs)
    dL = DataLoader()
    wdic = WordDictionary(w)
    tdic = TagDictionary()
    dL.readSRLData(sys.argv[1], wdic, tdic, False)
    batched_data = Process.create_batch_data(dL.sentences,cfg.BATCH_SIZE, wdic, tdic)
    print('saving dictionaries')

    save_word_to_idx(cfg.WORD_2_IDX_PATH,wdic)
    save_tag_to_idx(cfg.TAG_2_IDX_PATH,tdic)

    model = lstm_crf(len(wdic.word2idx), len(tdic.tag2idx), False)
    epoch = load_checkpoint(CRF_READ_GAN, model)
    model.eval();


    generator = Generator(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight(),tdic)
    generator.set_crf(model.crf)
    discriminator = Discriminator(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight())

    for param in model.parameters():
        param.requires_grad = False
    epoch = 0
    print("training model...")

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G)#, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D)#, betas=(opt.b1, opt.b2))
    criterion = myLoss()
    generator.train_start()
    discriminator.train()

    for ei in range(epoch+1, cfg.NUM_EPCH+epoch+1):
        G_loss_sum = 0
        D_loss_sum = 0
        timer = time.time()
        adjust_learning_rate(LR_G, optimizer_G, ei)
        adjust_learning_rate(LR_D, optimizer_D, ei)
        y_real = ones(1)
        count = 0
        for x, f, y in batched_data:

            prob = np.random.uniform(0,1)
            if prob > 0.5:
                count = count + 1
                noise = model.decode(x,f)
                noise = [ns[1:] for ns in noise]# the first one is SOS
                noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.

                #print('shape of noise {} {}'.format(noise.shape, x.shape))
                #print(f)
                result = generator(x,f,noise,y)
                result = [rs[1:] for rs in result]
                print('noise len before {}'.format(len(result[0])))
                result = Process.process_noise(result)#pad it again, make tensor.
                print('noise len before {}'.format(result.shape))
                y = y[:,1:]
                #print('result shape:{};  actual shape:{}'.format(result.shape, y.shape))
                validity_gen0 = discriminator(x,f,result)
                validity_gen = discriminator(x,f,result.detach())# act sen, syn tag
                validity_act = discriminator(x,f,y)# act sen, act tag

                ##-- find the hardest negative----##
                validity_wrong = discriminator(x,f,y[0].expand_as(y)).squeeze(-1)
                #print('worng score:{}'.format(validity_wrong))
                for i in range(1,y.shape[0]):
                    vw = discriminator(x,f,y[i].expand_as(y)).squeeze(-1)
                    #print(vw)
                    validity_wrong = torch.cat((validity_wrong, vw),0) #act tag, WRONG sen

                validity_wrong = validity_wrong.view(y.shape[0],-1)

                validity_wrong = validity_wrong.t()
                mask = eye(validity_wrong .size(0)) > .5
                validity_wrong = validity_wrong .masked_fill_(mask, 0)
                #print(sc_gen);
                validity_wrong = validity_wrong.max(1)[0]
                validity_wrong = validity_wrong.unsqueeze(-1)
                if y_real.shape != validity_act.shape:
                    with torch.no_grad():
                        y_real = ones(validity_act.shape)
                        y_fake = zeros(validity_gen.shape)
                #print('val score:{}'.format(validity_gen))
                #print('act score:{}'.format(validity_act))
                #print('worng score:{}'.format(validity_wrong))

                D_fake_loss = BCE_loss(validity_gen, y_fake)
                D_hn_loss = BCE_loss(validity_wrong, y_fake)
                D_real_loss = BCE_loss(validity_act, y_real)

                #D_train_loss = D_real_loss + D_fake_loss
                D_train_loss = D_real_loss + 0.5*( D_fake_loss + D_hn_loss )
                G_train_loss = BCE_loss(validity_gen0, y_real)
                print('error  type ={} {}'.format(D_train_loss.type(), D_train_loss.item))



                optimizer_D.zero_grad()
                D_train_loss.backward(retain_graph=True)      # reusing computational graph
                optimizer_D.step()


                optimizer_G.zero_grad()
                G_train_loss.backward()
                optimizer_G.step()

                G_loss_sum+= G_train_loss.detach()
                D_loss_sum+= D_train_loss.detach()

                if D_loss_sum.requires_grad == True:
                    print('err')
                else:
                    print('no err')
        timer = time.time()- timer
        if count >0:
            G_loss_sum/= count
            D_loss_sum/= count

        print('loss=={}, {}'.format(G_loss_sum, D_loss_sum))
        if ei % cfg.SAVE_EVERY == 0 or ei == epoch + cfg.NUM_EPCH:
            save_checkpoint(filename+"gen", generator, ei, G_loss_sum, timer)
            save_checkpoint(filename+"disc", discriminator, ei, G_loss_sum, timer)
