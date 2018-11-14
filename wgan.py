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

one = torch.FloatTensor([1])
mone = one * -1
if cfg.CUDA:
    one = one.cuda()
    mone = mone.cuda()

def train_disc(discriminator, generator, batched_data, model, D_optimizer):
    """ train the disc """
    freeze_net(generator)
    unfreeze_net(discriminator)
    timer = time.time()

    for ei in range(cfg.CRITIC_ITR):
        count = 0
        loss = 0
        for x, f, y in batched_data:
            count = count +1
            noise = model.decode(x,f)
            noise = [ns[1:] for ns in noise]# the first one is SOS. remove it to match with sentence dim
            noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.

            #print('shape of noise {} {}'.format(noise.shape, x.shape))
            #print(f)
            fake_data = generator(x,f,noise,y)
            fake_data = [rs[1:] for rs in fake_data]# removing SOS to match sentence dim
            fake_data  = Process.process_noise(fake_data )#pad it again, make tensor.
            y = y[:,1:] #removing SOS to match sentence len

            validity_gen = discriminator(x,f,fake_data.detach())
            disc_fake = validity_gen.mean()
            validity_act = discriminator(x,f,y)# act sen, act tag
            disc_real = validity_act.mean()
            '''
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
            sc_wrong = validity_wrong .masked_fill_(mask, 0)
            #print(sc_gen);
            sc_wrong = sc_wrong.max(1)[0]
            '''
            gradient_penalty = calc_gradient_penalty(discriminator, y, fake_data, x, f)
            D_optimizer.zero_grad()
            disc_cost = disc_fake - disc_real + gradient_penalty
            disc_cost.backward()
            w_dist = disc_fake  - disc_real
            D_train_loss.backward()
            D_optimizer.step()
            loss =  loss + disc_cost

        loss = loss/count

    timer = time.time() - timer

    unfreeze_net(generator)
    return discriminator, timer, loss

def train_generator(discriminator, generator, batched_data, model, G_optimizer):
    freeze_net(discriminator)
    unfreeze_net(generator)

    timer = time.time()

    for ei in range(cfg.GEN_ITR):
        count = 0
        loss = 0
        for x, f, y in batched_data:
            G_optimizer.zero_grad()
            noise = model.decode(x,f)
            noise = [ns[1:] for ns in noise]# the first one is SOS. remove it to match with sentence dim
            noise = Process.process_noise(noise)#pad it again, make same len as sentence. make tensor.
            noise.requires_grad_(True)
            #print('shape of noise {} {}'.format(noise.shape, x.shape))
            #print(f)
            fake_data = generator(x,f,noise,y)
            fake_data = [rs[1:] for rs in fake_data]# removing SOS to match sentence dim
            fake_data  = Process.process_noise(fake_data )
            validity_gen = discriminator(x,f,fake_data)
            gen_cost = validity_gen.mean()
            gen_cost.backward(mone)
            gen_cost = -gen_cost
            G_optimizer.step()
            count = count + 1
            loss = loss+ gen_cost
        loss = loss/count


    timer = time.time() - timer
    unfreeze_net(discriminator)
    return generator, timer, loss

def calc_gradient_penalty(netD, real_data, fake_data,  x, f):
    LAMBDA = 10
    batch_size = real_data.shape[0]
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    #print(real_data)
    #print('real data shape {}'.format(real_data.shape))
    #print('alpha shape {}'.format(alpha.shape))
    #alpha = alpha.view(batch_size, real_data.shape[1], real_data.shape[2])
    #alpha = alpha.view(batch_size, 3, DIM, DIM)
    if cfg.CUDA:
        alpha = alpha.cuda()

    #fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    #print('alpha ={};  real_data={};  fake_data={}'.format(alpha.type(), real_data.type(),fake_data.type()))
    real_data = real_data.type(alpha.type())
    fake_data = fake_data.type(alpha.type())
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    if cfg.CUDA:
        interpolates = interpolates.cuda()

    point_five = 0.5* ones(interpolates.shape)
    if cfg.CUDA:
        point_five = point_five.cuda()
    interpolates = (interpolates + point_five ).long()
    interpolates.requires_grad_(True)

    disc_interpolates = netD( x, f, interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


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
        loss_sum = 0
        timer = time.time()
        adjust_learning_rate(LR_G, optimizer_G, ei)
        adjust_learning_rate(LR_D, optimizer_D, ei)
        start_time = time.time()

        discriminator, timer1, D_loss = train_disc(discriminator, generator, batched_data, model, optimizer_D)
        generator, timer2, G_loss = train_generator(discriminator, generator, batched_data, model, optimizer_G)

        save_checkpoint(filename+"w_gen", generator, ei, G_loss, timer2)
        save_checkpoint(filename+"disc", discriminator, ei, D_loss, timer1)

        print('done')
