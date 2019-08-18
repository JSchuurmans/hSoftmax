from __future__ import print_function
import time
import sys
import os

import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.autograd import Variable

from src.util.evaluator import Evaluator
from src.util.utils import *


class Trainer(object):
    
    def __init__(self, model, optimizer, result_path, model_name, tag_to_id, usedataset,
                 eval_every=1, usecuda = True):
        self.model = model
        self.optimizer = optimizer
        self.eval_every = eval_every
        self.result_path = result_path
        # self.model_name = os.path.join(result_path, model_name)
        self.usecuda = usecuda
        self.tagset_size = len(tag_to_id)
        
        self.evaluator = Evaluator(result_path).evaluate

        self.model_name = model_name
    
    def adjust_learning_rate(self, optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
    def train_model(self, num_epochs, train_data, test_data, learning_rate, checkpoint_path, 
                    eval_train=True, plot_every=20, adjust_lr=False, batch_size = 50,
                    sched=None):

        losses = []
        loss = 0.0
        best_test_A = -1.0
        best_train_A = -1.0
        best_test_P = -1.0
        best_train_P = -1.0
        best_test_R = -1.0
        best_train_R = -1.0
        best_test_F = -1.0
        best_train_F = -1.0
        all_A=[[0,0]]
        all_P = [[0,0]]
        all_R = [[0,0]]
        all_F = [[0,0]]
        count = 0
        batch_count = 0
        
        self.model.train(True)
        if sched != None:
            scheduler = sched(self.optimizer, T_max=num_epochs)

        for epoch in range(1, num_epochs+1):
            t=time.time()
            if sched != None:
                scheduler.step()
            
            train_batches = create_batches(train_data, batch_size= batch_size, order='random')
            n_batches = len(train_batches)
            
            for i, index in enumerate(np.random.permutation(len(train_batches))): 
                
                data = train_batches[index]
                self.model.zero_grad()

                words = data['words']
                tags = data['tags']
                
                if self.usecuda:
                    words = Variable(torch.LongTensor(words)).cuda()
                    tags = Variable(torch.LongTensor(tags)).cuda()
                else:
                    words = Variable(torch.LongTensor(words))
                    tags = Variable(torch.LongTensor(tags))
                
                wordslen = data['wordslen']
                
                # print(type(self.tagset_size))

                if self.model_name[-4:] == 'LSTM':
                    score = self.model(words, tags, wordslen) #, n_batches)
                #,   usecuda=self.usecuda)
                elif self.model_name[-2:] == 'BB':
                    score = self.model(words, tags, wordslen, n_batches)
                
                loss += score.item()/len(wordslen) #TODO pytorch v.3: score.data[0]/len(wordslen)
                score.backward()
                
                nn.utils.clip_grad_norm(self.model.parameters(), 5.0)
                self.optimizer.step()
                
                count += 1
                batch_count += len(wordslen)
                
                if count % plot_every == 0:
                    loss /= plot_every
                    print(batch_count, ': ', loss)
                    if losses == []:
                        losses.append(loss)
                    losses.append(loss)
                    loss = 0.0
              
            # move this   
            # def on_train_begin(self):
            #     cycle_iter = 0
            #     cycle_count = 0

            # def calc_lr(self, init_lrs):
            #     if iteration < nb/20:
            #         cycle_iter += 1
            #         return init_lrs/100

            #     cos_out = np.cos(np.pi*cycle_iter/nb)+1
            #     cycle_count +=1
            #     if cycle_iter == nb:
            #         cycle_iter = 0
            #         nb *= cycle_mult
            #         if on_cylce_end: on_cycle_end(self, cycle_count)
            #         cycle_count += 1
            #     return init_lrs / 2* cos_out


            if adjust_lr:
                
                self.adjust_learning_rate(self.optimizer, lr=learning_rate/(1+0.05*float(epoch)/(num_epochs+1)))
                print(lr)
                print(param_group['lr'])

            if epoch%self.eval_every==0:
                
                self.model.train(False)
                if eval_train:
                    best_train_A, new_train_A, best_train_P, new_train_P, best_train_R, new_train_R, best_train_F, new_train_F, _ = self.evaluator(self.model, train_data, 
                                                                    best_train_A, best_train_P, best_train_R, best_train_F)
                else:
                    best_train_A, new_train_A, best_train_P, new_train_P, best_train_R, new_train_R, best_train_F, new_train_F, _ = 0, 0, 0, 0, 0, 0, 0, 0, 0
                best_test_A, new_test_A, best_test_P, new_test_P, best_test_R, new_test_R,best_test_F, new_test_F, save = self.evaluator(self.model, test_data, 
                                                                best_test_A, best_test_P, best_test_R, best_test_F)
                # TODO dont save the best model based on test score
                if save:
                    print ('*'*80)
                    print ('Saving Best Weights')
                    print ('*'*80)
                    torch.save(self.model, os.path.join(checkpoint_path, 'modelweights'))
                    
                sys.stdout.flush()
                all_A.append([new_train_A, new_test_A])
                all_P.append([new_train_P, new_test_P])
                all_R.append([new_train_R, new_test_R])
                all_F.append([new_train_F, new_test_F])
                self.model.train(True)

            print('*'*80)
            print('Epoch %d Complete: Time Taken %d' %(epoch ,time.time() - t))

        return losses, all_A, all_P, all_R, all_F