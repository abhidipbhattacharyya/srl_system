import torch
import torch.nn as nn
from configuration.config import *
#code from https://github.com/threelittlemonkeys/lstm-crf-pytorch/blob/master/model.py


PAD = "<PAD>" # padding
EOS = "<EOS>" # end of sequence
SOS = "<SOS>" # start of sequence
UNK = "<UNK>" # unknown token

PAD_IDX = 0
EOS_IDX = 1
SOS_IDX = 2
UNK_IDX = 3

torch.manual_seed(1)
CUDA = torch.cuda.is_available()

class lstm_crf(nn.Module):
    def __init__(self, vocab_size, num_tags, pretrained = False, pretrained_weight= None):#embed_size, hidden_size, num_dir, num_layers, bidirectional, dropout,
        super().__init__()

        # architecture
        self.lstm = lstm(vocab_size, num_tags, pretrained, pretrained_weight)
        self.crf = crf(num_tags)

        if CUDA:
            self = self.cuda()

    def forward(self, x,f, y0): # for training
        mask = x.data.gt(0).float()
        y = self.lstm(x,f, mask)
        Z = self.crf.forward(y, mask)
        score = self.crf.score(y, y0, mask)
        return Z - score # NLL loss

    def decode(self, x,f): # for prediction
        mask = x.data.gt(0).float()
        y = self.lstm(x,f, mask)
        return self.crf.decode(y, mask)

class lstm(nn.Module):
    def __init__(self, vocab_size, num_tags, pretrained = False,pretrained_weight= None ):
        super().__init__()
        self.embed_size= EMBED_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.num_dir = 1
        self.num_layers = 1
        self.bidirectional = False
        self.dropout = DROPOUT
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        # lstm input size == embed size and number of feature. make this one dynamic
        self.lstm_input_size= self.embed_size + 1 #number of features


        # architecture
        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx = PAD_IDX)
        if pretrained == True:
            self.embed.weight.data = Tensor(pretrained_weight)#, dtype=torch.float
        self.lstms = list()
        if CUDA:
            self.lstms = self.lstms.cuda()
        self.lstms.append(nn.LSTM(
            input_size = self.lstm_input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            bias = True,
            batch_first = True,
            dropout = self.dropout,
            bidirectional = self.bidirectional
        ))

        for i in range(1, SATCK_LAYERS):
            print("called")
            self.lstms.append(nn.LSTM(
                input_size = self.hidden_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                bias = True,
                batch_first = True,
                dropout = self.dropout,
                bidirectional = self.bidirectional
            ))
        self.out = nn.Linear(self.hidden_size, self.num_tags) # LSTM output to tag

    def init_hidden(self, batchsize): # initialize hidden states
        h = zeros(self.num_layers * self.num_dir, batchsize, self.hidden_size ) # hidden states
        c = zeros(self.num_layers * self.num_dir, batchsize, self.hidden_size ) # cell states
        #print('shape of h: {} {} {} '.format(self.num_layers * self.num_dir,self.hidden_size // self.num_dir, h.shape))
        #print('shape of c: {} '.format(c.shape))
        return (h, c)

    def forward(self, x, f, mask):
        #print('here')
        self.hidden = self.init_hidden(x.shape[0])
        #print('shape before embed: {} '.format(x.shape))
        embed = self.embed(x)
        #print('shape after embed: {} {} '.format(embed.type(),embed.shape))

        f1 = f[:,0]#take the feature of interest
        f1 = f1.unsqueeze(-1)# make it proper shape batch,seq,features
        f1 = f1.type(embed.type())
        #print('shape of feature: {} {} '.format(f1.type(),f1.shape))
        embed = cat((embed,f1),2)
        #print('shape after embed: {} {}'.format(embed.type(), embed.shape))

        embed = nn.utils.rnn.pack_padded_sequence(embed, mask.sum(1).int(), batch_first = True)
        h, _ = self.lstms[0](embed, self.hidden)
        h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)
        #print(h.shape)
        for i in range(1,SATCK_LAYERS):
            inv_idx = torch.arange(h.size(1)-1, -1, -1).long()
            inv_h_f = h.index_select(1, inv_idx)
            #inv_h_f = nn.utils.rnn.pack_padded_sequence(inv_h_f, mask.sum(1).int(),batch_first = True)
            h, _ = self.lstms[i](inv_h_f)
            #print('@{} lstm- shape-{} '.format(i,h.shape))
            #h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first = True)

        #print('shape before linear: {} '.format(h.shape))
        y = self.out(h)
        y *= mask.unsqueeze(-1).expand_as(y)
        return y

class crf(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[SOS_IDX, :] = -10000. # no transition to SOS
        self.trans.data[:, EOS_IDX] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, PAD_IDX] = -10000. # no transition from PAD except to PAD
        self.trans.data[PAD_IDX, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[PAD_IDX, EOS_IDX] = 0.
        self.trans.data[PAD_IDX, PAD_IDX] = 0.

    def forward(self, y, mask): # forward algorithm
        # initialize forward variables in log space
        #print('shape crf: {} '.format(y.shape[0]))
        batch_size_this = y.shape[0]
        score = Tensor(batch_size_this, self.num_tags).fill_(-10000.)
        score[:, SOS_IDX] = 0.
        for t in range(y.size(1)): # iterate through the sequence
            mask_t = mask[:, t].unsqueeze(-1).expand_as(score)
            score_t = score.unsqueeze(1).expand(-1, *self.trans.size())
            emit = y[:, t].unsqueeze(-1).expand_as(score_t)
            trans = self.trans.unsqueeze(0).expand_as(score_t)
            score_t = log_sum_exp(score_t + emit + trans)
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score)
        return score # partition function

    def score(self, y, y0, mask): # calculate the score of a given sequence
        batch_size_this = y.shape[0]
        score = Tensor(batch_size_this).fill_(0.)
        y0 = torch.cat([LongTensor(batch_size_this, 1).fill_(SOS_IDX), y0], 1)
        for t in range(y.size(1)): # iterate through the sequence
            mask_t = mask[:, t]
            emit = torch.cat([y[b, t, y0[b, t + 1]].unsqueeze(0) for b in range(batch_size_this)])
            trans = torch.cat([self.trans[seq[t + 1], seq[t]].unsqueeze(0) for seq in y0]) * mask_t
            score = score + emit + trans
        return score

    def decode(self, y, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = LongTensor()
        batch_size_this = y.shape[0]
        score = Tensor(batch_size_this, self.num_tags).fill_(-10000.)
        score[:, SOS_IDX] = 0.

        for t in range(y.size(1)): # iterate through the sequence
            # backpointers and viterbi variables at this timestep
            bptr_t = LongTensor()
            score_t = Tensor()
            for i in range(self.num_tags): # for each next tag
                m = [e.unsqueeze(1) for e in torch.max(score + self.trans[i], 1)]
                bptr_t = torch.cat((bptr_t, m[1]), 1) # best previous tags
                score_t = torch.cat((score_t, m[0]), 1) # best transition scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t + y[:, t] # plus emission scores
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size_this):
            x = best_tag[b] # best tag
            l = int(scalar(mask[b].sum()))
            for bptr_t in reversed(bptr[b][:l]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()

        return best_path

def Tensor(*args):
    x = torch.Tensor(*args)
    return x.cuda() if CUDA else x

def LongTensor(*args):
    x = torch.LongTensor(*args)
    return x.cuda() if CUDA else x

def randn(*args):
    x = torch.randn(*args)
    return x.cuda() if CUDA else x

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if CUDA else x

def cat(*args):
    x = torch.cat(*args)
    return x.cuda() if CUDA else x

def scalar(x):
    return x.view(-1).data.tolist()[0]

def argmax(x): # for 1D tensor
    return scalar(torch.max(x, 0)[1])

def log_sum_exp(x):
    max_score, _ = torch.max(x, -1)
    max_score_broadcast = max_score.unsqueeze(-1).expand_as(x)
    return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast), -1))
