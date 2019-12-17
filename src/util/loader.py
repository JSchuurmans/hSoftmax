from __future__ import print_function
import os
import torch
torch.manual_seed(0)
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

from src.util.utils import *

import codecs
# import cPickle
import itertools
from collections import Counter

import pandas as pd

import random

random.seed(359244)


class Loader(object):
    
    def __init__(self):
        pass
    
    def word_mapping(self, dataset):
        
        words = [[x.lower() for x in s[0].split()] for s in dataset]
        dico = create_dico(words)

        dico['<PAD>'] = 10000001
        dico['<UNK>'] = 10000000
        dico = {k:v for k,v in dico.items() if v>=2}
        word_to_id, id_to_word = create_mapping(dico)

        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        return dico, word_to_id, id_to_word
    
    def load_trec(self, datapath, pretrained, word_dim = 300, hier=False, data_name='trec'):
        
        trainpath = os.path.join(datapath, 'train_5500.label')
        testpath = os.path.join(datapath, 'TREC_10.label')
        
        train_data = []
        with open (trainpath, encoding="ISO-8859-1") as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                cat = content[0].split(':')[1]
                tag = content[0]
                train_data.append((sentence, tag))        
        
        test_data = []
        with open (testpath) as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                cat = content[0].split(':')[1]
                tag = content[0]
                test_data.append((sentence, tag))  
                
        dico_words_train = self.word_mapping(train_data)[0]
        
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data, hier=hier, name=data_name)
        
        print(dico_tags)
        print('*')
        print(tag_to_id)
        print('*')
        print(id_to_tag)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)
        
           
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                     
        return train_data_final, test_data_final, mappings

    def load_20ng(self, datapath, pretrained, word_dim = 300, hier=False, data_name='20ng'):
        
        trainpath = os.path.join(datapath, '20ng-train-all-terms.txt')
        testpath = os.path.join(datapath, '20ng-test-all-terms.txt')
        
        train_data = []
        with open (trainpath) as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                # cat = content[0].split('.')[1]
                tag = content[0].split('\t')[0]
                train_data.append((sentence, tag))        
        
        test_data = []
        with open (testpath) as f:
            for line in f:
                content = line.strip().split(' ')
                sentence = ' '.join(content[1:])
                # cat = content[0].split('.')[1]
                tag = content[0].split('\t')[0]
                test_data.append((sentence, tag))  
                
        dico_words_train = self.word_mapping(train_data)[0]
        
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data, hier=hier, name=data_name)
        
        print(dico_tags)
        print('*')
        print(tag_to_id)
        print('*')
        print(id_to_tag)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)
        
           
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                     
        return train_data_final, test_data_final, mappings

    def load_r8(self, datapath, pretrained, word_dim = 300, hier=False, data_name='r8'):
        if data_name=='r8':
            trainpath = os.path.join(datapath, 'r8-train-all-terms.txt')
            testpath = os.path.join(datapath, 'r8-test-all-terms.txt')
        elif data_name=='r52':
            trainpath = os.path.join(datapath, 'r52-train-all-terms.txt')
            testpath = os.path.join(datapath, 'r52-test-all-terms.txt')

        train_data = []
        with open (trainpath) as f:
            for line in f:
                content = line.strip().split('\t')
                sentence = content[1]
                # cat = content[0].split('.')[1]
                tag = content[0]
                train_data.append((sentence, tag))        
        
        test_data = []
        with open (testpath) as f:
            for line in f:
                content = line.strip().split('\t')
                sentence = content[1]
                # cat = content[0].split('.')[1]
                tag = content[0]
                test_data.append((sentence, tag))  
                
        dico_words_train = self.word_mapping(train_data)[0]
        
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data, hier=hier, name=data_name)
        
        print(dico_tags)
        print('*')
        print(tag_to_id)
        print('*')
        print(id_to_tag)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)
        
           
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                     
        return train_data_final, test_data_final, mappings

    def load_ya(self, datapath, pretrained, word_dim = 300, hier=False, data_name='ya', train_frac=.8):
        
        tags = os.listdir(datapath)
        
        train_data = []
        test_data = []
        for tag in tags:
            file_path = os.path.join(datapath,tag)
            
            if data_name == 'ya_16':
                tag = tag.split('.')[0]
            tag_data = []
            skip_next = False
            
            with open(file_path) as f:
                for line in f:
                    if line[0] =='<' or skip_next:
                        skip_next = False
                        continue
                    else:
                        skip_next = True
                        line.strip().replace('</TEXT>', '').replace('\n', ' ')
                    tag_data.append((line,tag))
            train_sample = random.sample(tag_data, int(len(tag_data)*train_frac))
            test_sample = list(set(tag_data)-set(train_sample))

            train_data.extend(train_sample)
            test_data.extend(test_sample)

                
        dico_words_train = self.word_mapping(train_data)[0]
        
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data, hier=hier, name=data_name)
        
        print(dico_tags)
        print('*')
        print(tag_to_id)
        print('*')
        print(id_to_tag)

        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)
        
           
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))


        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                     
        return train_data_final, test_data_final, mappings
    
    def load_mareview(self, datapath, pretrained, word_dim = 100):
        
        trainpospath = os.path.join(datapath, 'train-rt-polarity.pos')
        trainnegpath = os.path.join(datapath, 'train-rt-polarity.neg')
        
        testpospath = os.path.join(datapath, 'test-rt-polarity.pos')
        testnegpath = os.path.join(datapath, 'test-rt-polarity.neg')
        
        train_pos_data = []
        with open (trainpospath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 1
                train_pos_data.append((sentence, tag))        
        
        train_neg_data = []
        with open (trainnegpath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 0
                train_neg_data.append((sentence, tag))
                
        test_pos_data = []
        with open (testpospath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 1
                test_pos_data.append((sentence, tag))        
        
        test_neg_data = []
        with open (testnegpath) as f:
            for line in f:
                sentence = re.sub(r'[^\x00-\x7F]+',' ', line.strip())
                tag = 0
                test_neg_data.append((sentence, tag))
                
        train_data = train_pos_data + train_neg_data
        test_data = test_pos_data + test_neg_data
        
        dico_words_train = self.word_mapping(train_data)[0]
                
        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in test_data])
                ) if not all_embedding else None)
        
        dico_tags, tag_to_id, id_to_tag = tag_mapping(train_data)
        
        train_data_final = prepare_dataset(train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(test_data, word_to_id, tag_to_id)
        
        
        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        for w in word_to_id:
            if w in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w]
            elif w.lower() in all_word_embeds:
                word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }
                
        return train_data_final, test_data_final, mappings

    def load_pickle(self, datapath, pretrained, word_dim = 300, 
                    train='Braun_train.pickle', test='Braun_test.pickle',
                    label='intent', word_vectors=None,
                    tags=None):#, valid=False):
        
        trainpath = os.path.join(datapath, train)
        testpath = os.path.join(datapath, test)

        train_data = pd.read_pickle(trainpath)
        test_data = pd.read_pickle(testpath)

        list_train_data = train_data['utterance'].tolist()
        list_test_data = test_data['utterance'].tolist()
        
        train_label = train_data[label].tolist()
        test_label = test_data[label].tolist()

        labeled_train_data = []
        for i in range(len(train_label)):
            sentence = list_train_data[i]
            tag = train_label[i]
            labeled_train_data.append((sentence,tag))

        labeled_test_data = []
        for i in range(len(test_label)):
            sentence = list_test_data[i]
            tag = test_label[i]
            labeled_test_data.append((sentence,tag))            


        dico_words_train = self.word_mapping(labeled_train_data)[0]

        all_embedding = False
        dico_words, word_to_id, id_to_word = augment_with_pretrained(
                dico_words_train.copy(),
                pretrained,
                list(itertools.chain.from_iterable(
                    [[w.lower() for w in s[0].split()] for s in labeled_test_data])
                ) if not all_embedding else None)
        
        if tags is None:
            dico_tags, tag_to_id, id_to_tag = tag_mapping(labeled_train_data)
        else:
            tag_to_id, id_to_tag = tags

        train_data_final = prepare_dataset(labeled_train_data, word_to_id, tag_to_id)
        test_data_final = prepare_dataset(labeled_test_data, word_to_id, tag_to_id)
        
        word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(word_to_id), word_dim))

        all_word_embeds = {}
        for i, line in enumerate(codecs.open(pretrained, 'r', 'utf-8')):
            s = line.strip().split()
            if len(s) == word_dim + 1:
                all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

        words_not_in_vocab = set()
        if word_vectors is not None:
            for w in word_to_id:
                try:
                    if w in all_word_embeds:
                        word_embeds[word_to_id[w]] = word_vectors[w]
                    elif w.lower() in all_word_embeds:
                        word_embeds[word_to_id[w]] = word_vectors[w.lower()]
                except:
                    words_not_in_vocab.add(w)

                
        else:
            for w in word_to_id:
                if w in all_word_embeds:
                    word_embeds[word_to_id[w]] = all_word_embeds[w]
                elif w.lower() in all_word_embeds:
                    word_embeds[word_to_id[w]] = all_word_embeds[w.lower()]

        print('Loaded %i pretrained embeddings.' % len(all_word_embeds))

        mappings = {
            'word_to_id': word_to_id,
            'tag_to_id': tag_to_id,
            'id_to_tag': id_to_tag,
            'word_embeds': word_embeds
        }

        return train_data_final, test_data_final, mappings