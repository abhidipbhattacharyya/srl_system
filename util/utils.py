import re
from model import *
import torch
from reader.Dictionary import *
## code from https://github.com/threelittlemonkeys/lstm-crf-pytorch/blob/master/utils.py
def normalize(x):
    # x = re.sub("[^ a-zA-Z0-9\uAC00-\uD7A3]+", " ", x)
    # x = re.sub("[\u3040-\u30FF]+", "\u3042", x) # convert Hiragana and Katakana to あ
    # x = re.sub("[\u4E00-\u9FFF]+", "\u6F22", x) # convert CJK unified ideographs to 漢
    x = re.sub("\s+", " ", x)
    x = re.sub("^ | $", "", x)
    x = x.lower()
    return x

def tokenize(x, unit):
    x = normalize(x)
    if unit == "char":
        x = re.sub(" ", "", x)
        return list(x)
    if unit == "word":
        return x.split(" ")

def load_tag_to_idx(filename):
    print("loading tag_to_idx...")
    tag_to_idx = {}
    idx_to_tag = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        tag_to_idx[line] = len(tag_to_idx)
    fo.close()
    tdic = TagDictionary()
    #idx_to_tag = [tag for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1])]
    for key, value in tag_to_idx.items():
        value = int(value)
        idx_to_tag[value] = key
    tdic.setTag2Index(tag_to_idx)
    tdic.setIndex2Tag(idx_to_tag)
    return tdic

def load_word_to_idx(filename):
    print("loading word_to_idx...")
    word_to_idx = {}
    fo = open(filename)
    for line in fo:
        line = line.strip()
        word_to_idx[line] = len(word_to_idx)
    fo.close()
    wdic = WordDictionary()
    wdic.setWord2Index(word_to_idx)
    return wdic

def load_checkpoint(filename, model = None):
    print("loading model...")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print("saved model: epoch = %d, loss = %f" % (checkpoint["epoch"], checkpoint["loss"]))
    return epoch

def save_checkpoint(filename, model, epoch, loss, time):
    print("epoch = %d, loss = %f, time = %f" % (epoch, loss, time))
    if filename and model:
        print("saving model...")
        checkpoint = {}
        checkpoint["state_dict"] = model.state_dict()
        checkpoint["epoch"] = epoch
        checkpoint["loss"] = loss
        torch.save(checkpoint, filename + ".epoch%d" % epoch)
        print("saved model at epoch %d" % epoch)

def f1(p, r):
    if p + r:
        return 2 * p * r / (p + r)
    return 0

def save_word_to_idx(fname, wdic):
    fo = open(fname, "w")
    word_to_idx = wdic.word2idx
    for word, _ in sorted(word_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % word)
    fo.close()

def save_tag_to_idx(fname,tdic):
    fo = open(fname, "w")
    tag_to_idx = tdic.tag2idx
    for tag, _ in sorted(tag_to_idx.items(), key = lambda x: x[1]):
        fo.write("%s\n" % tag)
    fo.close()

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True

def reduce_data(batched_data, prob =0.5):
    selected_data =[]
    for i in range(len(batched_data)):
        data = batched_data[i]
        chance = np.random.uniform(0,1)
        if chance >= prob:
            data.append(data)
    return selected_data
