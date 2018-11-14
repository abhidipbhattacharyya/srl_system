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

def create_noise(batched_data, model):
    noise_batch =[]
    for x, f, y in batched_data:
        noise = model.decode(x,f)
        noise = [ns[1:] for ns in noise]# the first one is SOS. remove it to match with sentence dim
        noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.
        noise_batch.append(noise)
    return noise_batch

def adjust_learning_rate(learning_rate, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 10 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def decay_learning_rate(learning_rate, optimizer, df):
    """Sets the learning rate to the initial LR
       decayed by 10 every 10 epochs"""
    lr = learning_rate * df
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train_disc(discriminator, generator, batched_data, model, D_optimizer, from_gen = True):
    """ train the disc """
    freeze_net(generator)
    freeze_net(model)
    unfreeze_net(discriminator)
    timer = time.time()
    for ei in range(cfg.CRITIC_ITR):
        d_loss_avg =0
        count = 0
        print('@disc epoch{}'.format(ei))
        for x, f, y in batched_data:
            prob = np.random.uniform(0,1)
            if prob > 0:
                count = count+1
                noise = model.decode(x,f)
                noise = [ns[1:] for ns in noise]# the first one is SOS. remove it to match with sentence dim
                noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.

                #print('shape of noise {} {}'.format(noise.shape, x.shape))
                #print(f)
                if from_gen:
                    fake_data = generator.decode(x,f,noise)
                    fake_data = [rs[1:] for rs in fake_data]# removing SOS to match sentence dim
                    fake_data  = Process.process_noise(fake_data )#pad it again, make tensor.
                    validity_gen = discriminator(x,f,fake_data.detach())
                else:
                    validity_gen = discriminator(x,f,noise.detach())
                y = y[:,1:] #removing SOS to match sentence len
                #validity_gen = discriminator(x,f,fake_data.detach())

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
                y_real = ones(validity_act.shape)
                y_fake = zeros(validity_gen.shape)

                D_fake_loss = BCE_loss(validity_gen, y_fake)
                D_hn_loss = BCE_loss(validity_wrong, y_fake)
                D_real_loss = BCE_loss(validity_act, y_real)

                #D_train_loss = D_real_loss + D_fake_loss
                D_train_loss = D_real_loss + 0.5*( D_fake_loss + D_hn_loss )
                D_optimizer.zero_grad()
                D_train_loss.backward()
                D_optimizer.step()

                d_loss_avg+= scalar(D_train_loss.detach())

        d_loss_avg/=count
        timer = time.time()- timer
        print('discriminator loss {}'.format(d_loss_avg))
    unfreeze_net(generator)
    return discriminator, timer, d_loss_avg, D_optimizer


def train_generator(discriminator, generator, batched_data, model, G_optimizer, epochs = cfg.GEN_ITR, isolation=False):
    #freeze_net(discriminator)
    unfreeze_net(generator)

    timer = time.time()
    for ei in range(epochs):
        G_loss_avg =0
        count = 0
        print('@generator epoch{}'.format(ei))
        for x, f, y in batched_data:
            prob = np.random.uniform(0,1)
            if prob > 0:
                count = count+1
                noise = model.decode(x,f)
                noise = [ns[1:] for ns in noise]# the first one is SOS. remove it to match with sentence dim
                noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.
                #print('shape of noise {} {}'.format(noise.shape, x.shape))
                #print(f)
                loss = generator(x,f, noise, y)
                fake_data = generator.decode(x,f,noise)
                fake_data = [rs[1:] for rs in fake_data]# removing SOS to match sentence dim
                fake_data  = Process.process_noise(fake_data )
                validity_gen = discriminator(x,f,fake_data)
                #y_real = ones(validity_gen.shape)
                #bc_loss = BCE_loss(validity_gen, y_real)
                validity_gen = 1-validity_gen.squeeze(-1)
                #print('shape of loss {}'.format(loss))
                #print('shape of score {}'.format(validity_gen))
                #print('shape of bc_loss {} '.format(bc_loss .shape))
                #G_train_loss = BCE_loss(validity_gen, y_real)+loss
                if isolation:
                    G_train_loss = torch.mean(loss)
                else:
                    G_train_loss = torch.dot(validity_gen , loss)/x.shape[0]

                #print('shape of mul score {}'.format(G_train_loss))

                #G_train_loss = Variable(G_train_loss, requires_grad = True)

                G_optimizer.zero_grad()
                G_train_loss.backward()
                G_optimizer.step()

                G_loss_avg+=scalar(G_train_loss.detach())

        G_loss_avg/= count
        print('generator loss {}'.format(G_loss_avg))
        timer = time.time()- timer
    unfreeze_net(discriminator)
    return generator, timer, G_loss_avg, G_optimizer

if __name__ == '__main__':

    # Binary Cross Entropy loss
    #argv[1] == training filename
    #argv[2] == model name to be save
    #argv[3] == GPU
    if len(sys.argv) > 3:
        torch.cuda.set_device(int(sys.argv[3]))
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
    #generator.set_crf(model.crf)
    discriminator = Discriminator(len(wdic.word2idx), len(tdic.tag2idx), True,wdic.getWeight())

    freeze_net(model)
    epoch = 0
    print("training model...")
    lr_g = cfg.LR_G
    lr_d = cfg.LR_D


    optimizer_G_ini = torch.optim.SGD(model.parameters(), lr = cfg.LEARNING_RATE, weight_decay = cfg.WEIGHT_DECAY)#rd_optimizer(generator.parameters(), lr=lr_g, dstep=25, drate=0.1)
    optimizer_D = rd_optimizer(discriminator.parameters(), lr=lr_d, dstep=5, drate = 0.1)#, betas=(opt.b1, opt.b2))

    generator.train_start()
    discriminator.train()

    discriminator, dtime, d_loss, optimizer_D = train_disc(discriminator, generator, batched_data, model, optimizer_D,False)
    generator, gtime, g_loss, _ = train_generator(discriminator, generator, batched_data, model, optimizer_G_ini,10,True)
    save_checkpoint(filename+"TTini_gen", generator, 10, g_loss, gtime)
    optimizer_G = rd_optimizer(generator.parameters(), lr=lr_g, dstep=20, drate=0.05)#, betas=(opt.b1, opt.b2))

    for ei in range(epoch+1, cfg.NUM_EPCH+epoch+1):
        print('@epoch=={}'.format(ei))
        #lr_g = decay_learning_rate(lr_g , optimizer_G, .05)
        #lr_d = decay_learning_rate(lr_d, optimizer_D, .1)
        discriminator, dtime, d_loss, optimizer_D = train_disc(discriminator, generator, batched_data, model, optimizer_D)
        print('lr====={}'.format(optimizer_D.learning_rate()))
        generator, gtime, g_loss, optimizer_G = train_generator(discriminator, generator, batched_data, model, optimizer_G)



        if ei % cfg.SAVE_EVERY == 0 or ei == epoch + cfg.NUM_EPCH:
            save_checkpoint(filename+"TT_gen", generator, ei, g_loss, gtime)
            save_checkpoint(filename+"TT_disc", discriminator, ei, d_loss, dtime)
    print("done")
