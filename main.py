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
DATASETS = ['trec','ya','r','20news']

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='trec', type=str, help=f'Dataset: {str(DATASETS)}')
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
bs = 32 # batch size
n_epochs = 5

log_dir = os.path.join('log','args.data',model_name,str(time()))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

loader = Loader()

if data == 'trec':
    dataset_path = os.path.join('data','trec')
    train_data, test_data, mappings = loader.load_trec(dataset_path,
                    word_path, wdim, hier=hier)
    n_cat = 5
    n_max = 22


word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

word_vocab_size = len(word_to_id)

if hier:
    output_size = 22*5
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

losses, _, all_P, all_R, all_F = trainer.train_model(n_epochs, train_data,
                                    test_data, lr, batch_size=bs,
                                    checkpoint_path=log_dir)

F1_train = all_F[-1][0]
F1_test = all_F[-1][1]

P_train = all_P[-1][0]
P_test = all_P[-1][1]

R_train = all_R[-1][0]
R_test = all_R[-1][1]

