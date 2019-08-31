# from collections import OrderedDict
import os
import argparse
from time import time

# import matplotlib.pyplot as plt 
import torch
import numpy as np 

from src.util import Trainer, Loader
from src.lstm import LSTM
from src.lstm import h_LSTM

# supported datasets
DATASETS = ['trec','20ng','r8','r52','ya','ya_16']

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='r8', type=str, help=f'Dataset: {str(DATASETS)}')
parser.add_argument('-hi', action='store_true')
parser.add_argument('-bi', action='store_true')

args = parser.parse_args()

data = args.data.lower()
if data not in DATASETS:
    raise NotImplementedError

print(f'Dataset: {args.data}')

model_name = 'hLSTM' if args.hi else 'LSTM'
print(f'Model: {model_name}')

# TODO work this out per dataset, per model
# model parameters
hier = args.hi # hierarchical softmax
dropout = .5
wdim = 300 # word embedding dimension {50, 100, 200, 300}
word_path = os.path.join('wordvectors',f'glove.6B.{wdim}d.txt')
hdim = 150  # hidden dimension
bidir = args.bi # bidirectional

# training parameters
lr = 0.001 # learning rate
bs = 10 # batch size
n_epochs = 10

log_dir = os.path.join('log',data,model_name,str(time()))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loader = Loader()

if data == 'trec':
    dataset_path = os.path.join('data','trec')
    train_data, test_data, mappings = loader.load_trec(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 6
    n_max = 22
elif data == '20ng':
    dataset_path = os.path.join('data','ng20')
    train_data, test_data, mappings = loader.load_20ng(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 6
    n_max = 5
elif data == 'r8':
    dataset_path = os.path.join('data','r8')
    train_data, test_data, mappings = loader.load_r8(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 4
    n_max = 2
elif data == 'r52':
    dataset_path = os.path.join('data','r52')
    train_data, test_data, mappings = loader.load_r8(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 4
    n_max = 31
    # n_cat = 7
    # n_max = 13
elif data == 'ya':
    dataset_path = os.path.join('data','ya','Yahoo','Yahoo.ESA_2')
    train_data, test_data, mappings = loader.load_ya(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 16
#     n_max = 30
    n_max = 29
elif data == 'ya_16':
    dataset_path = os.path.join('data','ya','Yahoo','Yahoo.ESA_16_2')
    train_data, test_data, mappings = loader.load_ya(dataset_path,
                    word_path, wdim, hier=hier, data_name=data)
    n_cat = 8
    n_max = 3
    
word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

word_vocab_size = len(word_to_id)

if hier:
    output_size = n_cat*n_max
    model = h_LSTM(word_vocab_size, wdim, hdim, output_size,
                pretrained = word_embeds, bidirectional = bidir,
                n_cat=n_cat, n_max=n_max)
else:
    output_size = len(tag_to_id)
    model = LSTM(word_vocab_size, wdim, hdim, output_size,
                pretrained = word_embeds, bidirectional = bidir)

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# train
trainer = Trainer(model, optimizer, log_dir, model_name,
                    tag_to_id, usedataset = data)

losses, all_A, all_P, all_R, all_F = trainer.train_model(n_epochs, train_data,
                                    test_data, lr, batch_size=bs,
                                    checkpoint_path=log_dir)


F1_train = all_F[-1][0]
F1_test = all_F[-1][1]

P_train = all_P[-1][0]
P_test = all_P[-1][1]

R_train = all_R[-1][0]
R_test = all_R[-1][1]

# TODO save model parameters
# TODO save metrics
# TODO plot loss curve