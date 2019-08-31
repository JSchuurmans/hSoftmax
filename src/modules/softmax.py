import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# import data
import math
from src.util.utils import *

class LinearDecoder(nn.Module):
    def __init__(self, nhid, ntoken):
        super(LinearDecoder, self).__init__()
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        decoded = self.decoder(inputs.view(inputs.size(0)*inputs.size(1), inputs.size(2)))
        return decoded.view(inputs.size(0), inputs.size(1), decoded.size(1))


class HierarchicalSoftmax(nn.Module):
    def __init__(self, ntokens, nhid, n_cat, n_max):
        super(HierarchicalSoftmax, self).__init__()

        # Parameters
        self.n_cat = n_cat# len(n_subclasses) # number of categories # self.ntokens = ntokens
        # self.n_subclasses = n_subclasses # list of nr of subclasses per category
        self.n_tokens_per_cat = n_max # max(n_subclasses)
        self.nhid = nhid

        # if ntokens_per_class is None:
        #     ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))
        # self.ntokens_per_class = ntokens_per_class

        # self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        # self.ntokens_actual = self.n_cat * self.n_tokens_per_cat

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.n_cat), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.n_cat), requires_grad=True)

        # self.layer_bottom_Ws = {}
        # self.layer_bottom_bs = {}
        # for i,n in enumerate(n_subclasses):
        #     self.layer_bottom_Ws[i] = nn.Parameter(torch.FloatTensor(self.nhid, n), requires_grad=True)
        #     self.layer_bottom_bs[i] = nn.Parameter(torch.FloatTensor(n), requires_grad=True)

        self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.n_cat, self.nhid, self.n_tokens_per_cat), requires_grad=True)
        self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.n_cat, self.n_tokens_per_cat), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        self.layer_bottom_W.data.uniform_(-initrange, initrange)
        self.layer_bottom_b.data.fill_(0)
        # for i,n in enumerate(n_subclasses):
        #     self.layer_bottom_Ws[i].data.uniform_(-initrange, initrange)
        #     self.layer_bottom_bs[i].data.fill_(0)


    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:
            # print('labels...')
            # print(labels)

            label_position_top = labels / self.n_tokens_per_cat # self.ntokens_per_class
            # print('label_pos_top')
            # print(label_position_top)

            # label_position_top =  
            label_position_bottom = labels % self.n_tokens_per_cat # self.ntokens_per_class

            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_W[label_position_top]), dim=1) + self.layer_bottom_b[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            # target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            # cuda
            target_probs = layer_top_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_top] * layer_bottom_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_bottom]
            # print(target_probs)
            return target_probs

        else:
            
            
            # print(f'input size: {inputs.size()}')
            # Remain to be implemented
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            # print(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]))
            # print(layer_top_probs[:,0])

            # # print(mul(layer_top_probs[:,0],tmp))
            # print(tmp)
            
            # word_probs = broad_ltp * tmp
            word_probs = layer_top_probs[:,0] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0])) # .transpose(0,1)
            # print(f'wordprob before concat: {word_probs.size()}')

            for i in range(1, self.n_cat):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_W[i]) + self.layer_bottom_b[i]))), dim=0)

            # print(f'after concat and t(): {torch.t(word_probs).size()}')
            return torch.t(word_probs)


class HierarchicalSoftmax2(nn.Module):
    def __init__(self, ntokens, nhid, n_subclasses, ntokens_per_class = None):
        super(HierarchicalSoftmax2, self).__init__()

        # Parameters
        self.n_cat = len(n_subclasses) # number of categories # self.ntokens = ntokens
        self.n_subclasses = n_subclasses # list of nr of subclasses per category
        self.n_tokens_per_cat = max(n_subclasses)
        self.nhid = nhid

        # if ntokens_per_class is None:
        #     ntokens_per_class = int(np.ceil(np.sqrt(ntokens)))
        # self.ntokens_per_class = ntokens_per_class

        # self.nclasses = int(np.ceil(self.ntokens * 1. / self.ntokens_per_class))
        # self.ntokens_actual = self.n_cat * self.n_tokens_per_cat

        self.layer_top_W = nn.Parameter(torch.FloatTensor(self.nhid, self.n_cat), requires_grad=True)
        self.layer_top_b = nn.Parameter(torch.FloatTensor(self.n_cat), requires_grad=True)

        self.layer_bottom_Ws = {}
        self.layer_bottom_bs = {}
        for i,n in enumerate(n_subclasses):
            self.layer_bottom_Ws[i] = nn.Parameter(torch.FloatTensor(self.nhid, n), requires_grad=True)
            self.layer_bottom_bs[i] = nn.Parameter(torch.FloatTensor(n), requires_grad=True)

        # self.layer_bottom_W = nn.Parameter(torch.FloatTensor(self.n_cat, self.nhid, self.n_tokens_per_cat), requires_grad=True)
        # self.layer_bottom_b = nn.Parameter(torch.FloatTensor(self.n_cat, self.n_tokens_per_cat), requires_grad=True)

        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

    def init_weights(self):

        initrange = 0.1
        self.layer_top_W.data.uniform_(-initrange, initrange)
        self.layer_top_b.data.fill_(0)
        # self.layer_bottom_W.data.uniform_(-initrange, initrange)
        # self.layer_bottom_b.data.fill_(0)
        for i,n in enumerate(self.n_subclasses):
            self.layer_bottom_Ws[i].data.uniform_(-initrange, initrange)
            self.layer_bottom_bs[i].data.fill_(0)


    def forward(self, inputs, labels = None):

        batch_size, d = inputs.size()

        if labels is not None:
            print('labels...'+str(type(labels)))
            print(labels)

            label_position_top = labels / self.n_tokens_per_cat # self.ntokens_per_class
            label_position_top = labels-labels # get array of zeros with dim of labels
            substracted_labels = labels
            
            for n in self.n_subclasses:
                substracted_labels -= n
                print(labels>0)
                label_position_top += labels>=0 # TODO fix TypeError
            # print('label_pos_top')
            # print(label_position_top)

            label_position_bottom = labels-labels
            for i,l in enumerate(labels):
                j = 0
                sub_label = l
                next_sub_label = sub_label-self.n_subclasses[j]
                while next_sub_label>=0:
                    sub_label = next_sub_label
                    j+=1
                    next_sub_label -= self.n_subclasses[j]
                label_position_bottom[i] = sub_label

            label_position_bottom = labels
            for n in self.n_subclasses:
                # state is boolean vector with length of labels, 
                #     initially true, 
                #     set to false once next is false
                #     once false, stay false
                state = np.array(len(labels)*[True])
                update = (label_position_bottom -n)>=0
                state = np.logical_or(state, update)
                label_position_bottom -= n*state


            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            # TODO if this does not work
            #   use torch.t(word_probs) and select probs with labels as index
            layer_bottom_logits = torch.squeeze(torch.bmm(torch.unsqueeze(inputs, dim=1), self.layer_bottom_Ws[label_position_top]), dim=1) + self.layer_bottom_bs[label_position_top]
            layer_bottom_probs = self.softmax(layer_bottom_logits)

            # target_probs = layer_top_probs[torch.arange(batch_size).long(), label_position_top] * layer_bottom_probs[torch.arange(batch_size).long(), label_position_bottom]
            # cuda
            target_probs = layer_top_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_top] * layer_bottom_probs[torch.arange(batch_size).type(torch.cuda.LongTensor), label_position_bottom]
            # print(target_probs)
            return target_probs

        else:
            
            
            # print(f'input size: {inputs.size()}')
            # Remain to be implemented
            layer_top_logits = torch.matmul(inputs, self.layer_top_W) + self.layer_top_b
            layer_top_probs = self.softmax(layer_top_logits)

            # print(self.softmax(torch.matmul(inputs, self.layer_bottom_W[0]) + self.layer_bottom_b[0]))
            # print(layer_top_probs[:,0])

            # # print(mul(layer_top_probs[:,0],tmp))
            # print(tmp)
            
            # word_probs = broad_ltp * tmp
            word_probs = layer_top_probs[:,0] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_Ws[0]) + self.layer_bottom_bs[0])) # .transpose(0,1)
            # print(f'wordprob before concat: {word_probs.size()}')

            for i in range(1, self.n_cat):
                word_probs = torch.cat((word_probs, layer_top_probs[:,i] * torch.t(self.softmax(torch.matmul(inputs, self.layer_bottom_Ws[i]) + self.layer_bottom_bs[i]))), dim=0)

            # print(f'after concat and t(): {torch.t(word_probs).size()}')
            return torch.t(word_probs)