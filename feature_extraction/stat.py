import numpy as np
import os
import sys
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from configuration.config import *
from reader.Dictionary import *
import numpy as np

class statistics:
    def __init__(self,wdic, tdic):
        self.wdic = wdic
        self.tdic = tdic
        self.p_matrix = [[0 for x in range(len(tdic.tag2idx)] for y in range(len(wdic.word2idx))]
    ## calculate the un-normalized probability
    ## P(tag|word)
    def set_p_matrix(self, p_matrix):
        self.p_matrix = p_matrix

    def create_prob_table(self, sentences):
        for sen in sentences:
            words = sen[1]
            tags = sen[2]
            for w,t in zip(words,tags):
                w = wdic.getIndex(w)
                t = tdic.getIndex(t)
                self.p_matrix[w][t] = self.p_matrix[w][t]+1

    def get_probable_tag(self, word):
        w = self.wdic.getIndex(word)
        tags_prob = np.array(self.p_matrix[w])
        arg_max = np.argmax(tags_prob)
        return self.tdic.getTag(arg_max)

    def get_probable_tag_sentence(self, words):
        tags = list()
        for word in words:
            tag = self.get_probable_tag(word)
            tags.append(tag)
        return tags

    def save_prob_table(self, fname):
        fo = open(fname, "w")
        for tags in self.p_matrix:
            for i in range(len(tags)):
                fo.write("%s" %str(tags[i]))
                if i <len(tags)-1:
                    fo.write(" ")
            fo.write("\n")

    def load_prob_table(self, fname):
        print("loading prob_table...")
        fo = open(fname)
        index = -1
        for line in fo:
            index = index+1
            t_counts =  line.strip().split(" ")
            t_counts = [int(t) for t in t_counts]
            self.p_matrix[index] = t_counts
