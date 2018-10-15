import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

class SRL_Genarator(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, pretrained_weight):
        super(SRL_Genarator, self).__init__()
        embed = nn.Embedding(vocab_size, embedding_dim)
        # pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
        embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

class SRL_Discriminator(nn.Module):
