import torch
import sys
import os

BATCH_SIZE = 64
EMBED_SIZE = 300
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0 #0.5
BIDIRECTIONAL = True
NUM_DIRS = 2 if BIDIRECTIONAL else 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
SAVE_EVERY = 10
SATCK_LAYERS = 4
USE_SE = False # if false only use end marker if true wil use start marker as well
OUTPUT_PATH = 'output/'
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token
SRL_CONLL_EVAL_SCRIPT  = 'scripts/run_eval.sh'
EMBEDDING = 'embedding/glove.6B.300d.txt'
TAG_EMBED_SIZE =5
MAX_SEQ_LEN = 141
PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3
CUDA = torch.cuda.is_available()
LR_G = 0.002
LR_D = 0.0002

CRF_READ_GAN = 'trained_model/model.epoch20'
TRAINDED_MODEL_PATH = 'trained_model'
WORD_2_IDX_PATH = 'trained_model/word_2_idx.txt'
NUM_EPCH = 20
GEN_ITR = 10
CRITIC_ITR = 5
TAG_2_IDX_PATH = 'trained_model/tag_2_idx.txt'
