# from collections import OrderedDict
import os
import argparse
from time import time
import pickle
import random
random.seed(359244)

# import matplotlib.pyplot as plt 
import torch
import numpy as np 
import pandas as pd

from src.util import Trainer, Loader
from src.lstm import LSTM
from src.lstm import h_LSTM

# supported datasets
DATASETS = ['trec','20ng','r8','r52','ya','ya_16']

parser = argparse.ArgumentParser()

parser.add_argument('--data', default='r8', type=str, help=f'Dataset: {str(DATASETS)}')
parser.add_argument('--hdim', default=100, type=int, help='Hidden dimension')
parser.add_argument('--epoch', default=10, type=int, help='Number of epochs')
parser.add_argument('--kfold', default=None, type=int, help='Do cross validation with k folds')
parser.add_argument('-hi', action='store_true')
parser.add_argument('-bi', action='store_true')


args = parser.parse_args()

param = dict()

data = args.data.lower()
if data not in DATASETS:
    raise NotImplementedError
param['data'] = data
print(f'Dataset: {args.data}')

model_name = 'LSTM'
model_name = 'Bi'+model_name if args.bi else model_name
model_name = 'h_'+model_name if args.hi else model_name
print(f'Model: {model_name}')
param['model'] = model_name

# TODO work this out per dataset, per model
# model parameters
hier = args.hi # hierarchical softmax
dropout = .5
wdim = 300 # word embedding dimension {50, 100, 200, 300}
word_path = os.path.join('wordvectors',f'glove.6B.{wdim}d.txt')
hdim = args.hdim  # hidden dimension
bidir = args.bi # bidirectional
param['hier'] = hier
param['dropout'] = dropout
param['wdim'] = wdim
param['hdim'] = hdim
param['bidir'] = bidir

# training parameters
lr = 0.001 # learning rate
bs = 10 # batch size
n_epochs = args.epoch
param['lr'] = lr
param['bs'] = bs
param['epochs'] = n_epochs

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

param['n_cat'] = n_cat
param['n_max'] = n_max
    
word_to_id = mappings['word_to_id']
tag_to_id = mappings['tag_to_id']
word_embeds = mappings['word_embeds']

word_vocab_size = len(word_to_id)

kfold = args.kfold
n = len(train_data)
if kfold is not None:
    log_dir_valid = os.path.join(log_dir,'valid')
    
    smpl = random.sample(range(n),n)
    n_k = int(n/kfold)
    start = 0
    end = n_k
    for k in range(kfold):
        
        log_dir_valid_fold = os.path.join(log_dir_valid,str(k))
        if not os.path.exists(log_dir_valid_fold):
            os.makedirs(log_dir_valid_fold)
        
        valid_test = [train_data[i] for i in smpl[start:end]]
        train_smpl = list(set(smpl)-set(smpl[start:end]))
        valid_train = [train_data[i] for i in train_smpl]
        valid_train = [i for i in valid_train]
        
        start = end
        end += n_k
        
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

        losses, all_A, all_P, all_R, all_F = trainer.train_model(n_epochs, valid_train, valid_test, lr, batch_size=bs, checkpoint_path=log_dir_valid_fold)

        A_train_all = [X[0] for X in all_A]
        A_test_all = [X[1] for X in all_A]

        F_train_all = [X[0] for X in all_F]
        F_test_all = [X[1] for X in all_F]

        P_train_all = [X[0] for X in all_P]
        P_test_all = [X[1] for X in all_P]

        R_train_all = [X[0] for X in all_R]
        R_test_all = [X[1] for X in all_R]

        train_metrics = pd.DataFrame({
            # 'losses': losses,
            'A': A_train_all,
            'F': F_train_all,
            'P': P_train_all,
            'R': R_train_all
        })
        train_metrics.to_csv(os.path.join(log_dir_valid_fold, 'train_metrics.csv'))
        train_metrics.to_pickle(os.path.join(log_dir_valid_fold, 'train_metrics.pkl'))

        test_metrics = pd.DataFrame({
            'A': A_test_all,
            'F': F_test_all,
            'P': P_test_all,
            'R': R_test_all
        })
        test_metrics.to_csv(os.path.join(log_dir_valid_fold,'test_metrics.csv'))
        test_metrics.to_pickle(os.path.join(log_dir_valid_fold,'test_metrics.pkl'))

        
else:
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

    A_train_all = [X[0] for X in all_A]
    A_test_all = [X[1] for X in all_A]

    F_train_all = [X[0] for X in all_F]
    F_test_all = [X[1] for X in all_F]

    P_train_all = [X[0] for X in all_P]
    P_test_all = [X[1] for X in all_P]

    R_train_all = [X[0] for X in all_R]
    R_test_all = [X[1] for X in all_R]

    train_metrics = pd.DataFrame({
        # 'losses': losses,
        'A': A_train_all,
        'F': F_train_all,
        'P': P_train_all,
        'R': R_train_all
    })
    train_metrics.to_csv(os.path.join(log_dir,'train_metrics.csv'))
    train_metrics.to_pickle(os.path.join(log_dir,'train_metrics.pkl'))

    test_metrics = pd.DataFrame({
        'A': A_test_all,
        'F': F_test_all,
        'P': P_test_all,
        'R': R_test_all
    })
    test_metrics.to_csv(os.path.join(log_dir,'test_metrics.csv'))
    test_metrics.to_pickle(os.path.join(log_dir,'test_metrics.pkl'))

with open(os.path.join(log_dir, 'param.pkl'), 'wb') as f:
    pickle.dump(param, f, protocol=pickle.HIGHEST_PROTOCOL)

# with open(os.path.join(log_dir, 'param.pkl'), 'rb') as f:
#     param = pickle.load(f)

# TODO loss curve