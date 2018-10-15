import os
import numpy as np
## TODO: create separate class for word and tagdictionary.
PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3
class WordDictionary:
    def __init__(self, pretrainedEmbedding = None):
        self.word2idx ={}
        self.idx2word = {}
        ## TODO: create pad and put it at 0
        self.word2idx[UNK ] = UNK_IDX
        self.word2idx[SOS] = SOS_IDX
        self.word2idx[EOS] = EOS_IDX
        self.word2idx[PAD] = PAD_IDX

        self.idx2word[UNK_IDX] = UNK
        self.idx2word[SOS_IDX] = SOS
        self.idx2word[EOS_IDX] = EOS
        self.idx2word[PAD_IDX] = PAD
        self.em = pretrainedEmbedding
        self.weights = list()
        self.dim = 300 ##TODO remove the hard coding
        # put the random for unknown, SOS, EOS, PAD. Else weight and index will not match
        self.weights.append(np.random.normal(0, 0.1, self.dim ))
        self.weights.append(np.random.normal(0, 0.1, self.dim ))
        self.weights.append(np.random.normal(0, 0.1, self.dim ))
        self.weights.append(np.random.normal(0, 0.1, self.dim ))

        self.pretrained_ini = False
        if pretrainedEmbedding is not None:
            self.pretrained_ini = True

    def add(self, word):
        if  word not in self.word2idx:
            if self.em is not None:
                ##self.idx2word[len(self.word2idx)] = word #order matters
                ##self.word2idx[word] = len(self.word2idx)
                if word in self.em: ## add a word if it is in embedding. Else unk
                    self.weights.append(self.em[word])
                    self.idx2word[len(self.word2idx)] = word #order matters
                    self.word2idx[word] = len(self.word2idx)
                #else:
                    #self.weights.append(np.random.normal(0, 0.1, self.dim ))

            else:
                self.idx2word[len(self.word2idx)] = word #order matters
                self.word2idx[word] = len(self.word2idx)

    def getIndex(self, word):
        #if word not in embedding:
            #return 0
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return UNK_IDX

    def setWord2Index(self, word2idx):
        self.word2idx = word2idx

    def getWord(self, index):
        return self.idx2word[index]

    def getWeight(self):
        return np.array(self.weights)

class TagDictionary:
    def __init__(self):
        self.tag2idx ={}
        self.idx2tag = {}
        ## TODO: create pad and put it at 0
        self.tag2idx[UNK ] = UNK_IDX
        self.tag2idx[SOS] = SOS_IDX
        self.tag2idx[EOS ] = EOS_IDX
        self.tag2idx[PAD] = PAD_IDX

        self.idx2tag[UNK_IDX] = UNK
        self.idx2tag[SOS_IDX] = SOS
        self.idx2tag[EOS_IDX] = EOS
        self.idx2tag[PAD_IDX] = PAD


    def add(self, tag):
        if  tag not in self.tag2idx:
            self.idx2tag[len(self.tag2idx)] = tag #order matters
            self.tag2idx[tag] = len(self.tag2idx)

    def getIndex(self, tag):
        #if word not in embedding:
            #return 0
        return self.tag2idx[tag]

    def getTag(self, index):
        return self.idx2tag[index]

    def setTag2Index(self, tag2idx):
        self.tag2idx = tag2idx

    def setIndex2Tag(self, idx2tag):
        self.idx2tag = idx2tag
