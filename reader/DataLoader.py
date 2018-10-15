import sys
import os
import numpy as np
from .Dictionary import WordDictionary, TagDictionary
from configuration.config import *


class DataLoader:
    def __init__(self):
        self.sentences= list()

    # for reading training
    def readSRLData(self, filename, wdic, tdic, dev = False):
        with open(filename, 'r') as f:
            lines = f.readlines()

        sen_index =0 # keep track which sentence. Need to retrive the actual oredering
        for line in lines:

            parts = line.strip().split("|||")
            tags = parts[1].strip().split(" ")
            leftpart = parts[0].lower().strip().split(" ")

            prop = int(leftpart[0])
            words = leftpart[1:]
            assert len(words) == len(tags)
            for w in words:
                wdic.add(w.lower())
            for t in tags:
                tdic.add(t)
            self.sentences.append((prop, words, tags,sen_index))
            sen_index = sen_index +1
        #if dev == False:# do not do it for develop set
        self.sentences.sort(key=lambda x: len(x[1]), reverse = True)
        print('maximum seq_length: {}'.format(len(self.sentences[0][1])))

    # for reading test data
    def readSRLTestData(self, filename, wdic, tdic):

        with open(filename, 'r') as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split("|||")
            tags = parts[1].strip().split(" ")
            leftpart = parts[0].lower().strip().split(" ")

            prop = int(leftpart[0])
            words = leftpart[1:]
            #assert len(words) == len(tags)
            word_test =[]
            tag_test =[]
            for w in words:
                if w in wdic:
                    word_test.append(w.lower())
                else:
                    word_test.append(UNK)

            if len(tags) > 1:
                for t in tags:
                    tag_test.append(t)

            self.sentences.append((prop, word_test, tag_test))
        self.sentences.sort(key=lambda x: len(x[1]), reverse = True)


if __name__ == '__main__':
    dL = DataLoader()
    dic = Dictionary()
    dL.readSRLData(sys.argv[1], dic)
    print(dL.sentences[0])
    print(dic.word2idx)
    print(dic.idx2word)
