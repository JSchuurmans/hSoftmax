import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F

# import models.neural_cls
from src.util import Initializer
from src.util import Loader
from src.modules import EncoderRNN

class LSTM(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, output_size, 
                 pretrained=None, n_layers = 1, bidirectional = True, dropout_p = 0.5, rnn_cell='lstm'):
        
        super(LSTM, self).__init__()
        
        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.rnn_cell = rnn_cell
        
        self.word_encoder = EncoderRNN(word_vocab_size, word_embedding_dim, word_hidden_dim, 
                                       n_layers = n_layers, bidirectional = bidirectional,
                                       rnn_cell=rnn_cell)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = 2*n_layers*word_hidden_dim if bidirectional and rnn_cell=='lstm' else n_layers*word_hidden_dim
        self.linear = nn.Linear(hidden_size, output_size)
        self.lossfunc = nn.CrossEntropyLoss()
        
    def forward(self, words, tags, wordslen): #, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        loss = self.lossfunc(output, tags)
        
        return loss
    
    def predict(self, words, wordslen, scoreonly=False, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        output = self.linear(word_features)
        
        scores = torch.max(F.softmax(output, dim =1), dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction


# TODO merge LSTM and h_LSTM and add arg 'hier' at LSTM
from src.modules.softmax import HierarchicalSoftmax
# from models.neural_cls.modules.softmax import HierarchicalSoftmax2

class h_LSTM(nn.Module):
    
    def __init__(self, word_vocab_size, word_embedding_dim, word_hidden_dim, output_size, 
                 pretrained=None, n_layers = 1, bidirectional = True, dropout_p = 0.5, rnn_cell='lstm',
                 n_cat = 5, n_max=22):
        
        super(h_LSTM, self).__init__()

        self.word_vocab_size = word_vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.word_hidden_dim = word_hidden_dim
        
        self.initializer = Initializer()
        self.loader = Loader()

        self.rnn_cell = rnn_cell
        
        self.word_encoder = EncoderRNN(word_vocab_size, word_embedding_dim, word_hidden_dim, 
                                       n_layers = n_layers, bidirectional = bidirectional,
                                       rnn_cell=rnn_cell)
        
        if pretrained is not None:
            self.word_encoder.embedding.weight = nn.Parameter(torch.FloatTensor(pretrained))
        
        self.dropout = nn.Dropout(p=dropout_p)
        
        hidden_size = 2*n_layers*word_hidden_dim if bidirectional and rnn_cell=='lstm' else n_layers*word_hidden_dim
        
        # hierachy
        self.decoder = HierarchicalSoftmax(ntokens= output_size, nhid = hidden_size, n_cat = n_cat, n_max=n_max)
                                            # n_subclasses = n_subclasses)
        
        self.lossfunc = nn.CrossEntropyLoss()
        
    def forward(self, words, tags, wordslen): #, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        # hierarchy
        output = self.decoder(word_features, tags)
        
        # loss = self.lossfunc(output, tags)
        loss = -torch.mean(torch.log(output))
        
        return loss
    
    def predict(self, words, wordslen, scoreonly=False, usecuda=True):
        
        batch_size, max_len = words.size()
        word_features = self.word_encoder(words, wordslen)
        word_features = self.dropout(word_features)
        # hierarchy
        output = self.decoder(word_features)
        
        #hierarchy
        scores = torch.max(output, dim=1)[0].data.cpu().numpy()
        if scoreonly:
            return scores
        
        prediction = torch.max(output, dim=1)[1].data.cpu().numpy().tolist()
        return scores, prediction