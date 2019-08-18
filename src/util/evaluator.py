import os
import codecs
import torch

from torch.autograd import Variable

from src.util.utils import *

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import confusion_matrix

class Evaluator(object):
    def __init__(self, result_path, usecuda=True): # model_name
        self.result_path = result_path
        # self.model_name = model_name
        self.usecuda = usecuda

    def evaluate(self, model, dataset, best_A, best_P, best_R, best_F, checkpoint_folder='.', batch_size = 32):
        
        predicted_ids = []
        ground_truth_ids = []
        
        save = False
        new_A = 0.0
        new_P = 0.0
        new_R = 0.0
        new_F = 0.0
        
        data_batches = create_batches(dataset, batch_size = batch_size)

        for data in data_batches:

            words = data['words']

            if self.usecuda:
                words = Variable(torch.LongTensor(words)).cuda()
            else:
                words = Variable(torch.LongTensor(words))

            wordslen = data['wordslen']
            
            _, out = model.predict(words, wordslen, usecuda = self.usecuda)         
            
            ground_truth_ids.extend(data['tags'])
            predicted_ids.extend(out)

        new_A = np.mean(np.array(ground_truth_ids) == np.array(predicted_ids))

        # Precision
        new_P = precision_score(ground_truth_ids, predicted_ids, average='macro')
        new_R = recall_score(ground_truth_ids, predicted_ids, average='macro')
        new_F = f1_score(ground_truth_ids, predicted_ids, average= 'macro')
        # Recall
        # F1    

        # TODO 
        if new_A > best_A:
            best_A = new_A
            save = True
        
        if new_P > best_P:
            best_P = new_P
            save = True

        if new_R > best_R:
            best_R = new_R
            save = True

        if new_F > best_F:
            best_F = new_F
            save = True

        print('*'*80)
        print('Accuracy: %f, Best Accuracy: %f' %(new_A, best_A))
        print('Precision: %f, Best Precision: %f' %(new_P, best_P))
        print('Recall: %f, Best Recall: %f' %(new_R, best_R))
        print('F1: %f, Best F1: %f' %(new_F, best_F))
        print('*'*80)
            
        return best_A, new_A, best_P, new_P, best_R, new_R, best_F, new_F, save