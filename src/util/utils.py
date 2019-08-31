from __future__ import print_function
import os
import re
import codecs
import copy
import numpy as np
np.random.seed(0)
import random
random.seed(0)
from collections import Counter
import torch
from data.trec.tag_mapping_trec import tag_to_id as tag_to_id_trec
from data.trec.tag_mapping_trec import id_to_tag as id_to_tag_trec
from data.ng20.tag_mapping_20ng import tag_to_id as tag_to_id_20ng
from data.ng20.tag_mapping_20ng import id_to_tag as id_to_tag_20ng
from data.r8.tag_mapping_r8 import tag_to_id as tag_to_id_r8
from data.r8.tag_mapping_r8 import id_to_tag as id_to_tag_r8
from data.r52.tag_mapping_r52_2 import tag_to_id as tag_to_id_r52
from data.r52.tag_mapping_r52_2 import id_to_tag as id_to_tag_r52
from data.ya.tag_mapping_ya_2 import tag_to_id as tag_to_id_ya
from data.ya.tag_mapping_ya_2 import id_to_tag as id_to_tag_ya
from data.ya.tag_mapping_ya_16_2 import tag_to_id as tag_to_id_ya_16
from data.ya.tag_mapping_ya_16_2 import id_to_tag as id_to_tag_ya_16

def create_dico(item_list):
    """
    Create a dictionary of items from a list of list of items.
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico

def create_mapping(dico):
    """
    Create a mapping (item to ID / ID to item) from a dictionary.
    Items are ordered by decreasing frequency.
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item

def augment_with_pretrained(dictionary, ext_emb_path, words):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    
    if words is None:
        for word in pretrained:
            if word not in dictionary:
                dictionary[word] = 0
    else:
        for word in words:
            if any(x in pretrained for x in [
                word,
                word.lower(),
                re.sub('\d', '0', word.lower())
            ]) and word not in dictionary:
                dictionary[word] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word

def tag_mapping(dataset, name='trec', hier=False):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    """
    tags = [s[1] for s in dataset]
    dico = Counter(tags)
    print("Found %i unique named entity tags" % len(dico))
    if not hier:        
        tag_to_id, id_to_tag = create_mapping(dico)
    else:
        if name=='trec':
            tag_to_id, id_to_tag = tag_to_id_trec, id_to_tag_trec
        elif name =='20ng':
            tag_to_id, id_to_tag = tag_to_id_20ng, id_to_tag_20ng
        elif name == 'r8':
            tag_to_id, id_to_tag = tag_to_id_r8, id_to_tag_r8
        elif name == 'r52':
            tag_to_id, id_to_tag = tag_to_id_r52, id_to_tag_r52
        elif name == 'ya':
            tag_to_id, id_to_tag = tag_to_id_ya, id_to_tag_ya
        elif name == 'ya_16':
            tag_to_id, id_to_tag = tag_to_id_ya_16, id_to_tag_ya_16
        else:
            raise NotImplementedError
    
    return dico, tag_to_id, id_to_tag

def prepare_dataset(dataset, word_to_id, tag_to_id):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    def f(x): return x.lower()
    data = []
    for s in dataset:
        str_words = [w for w in s[0].split()] # splitting of the sentence
        words = [word_to_id[f(w) if f(w) in word_to_id else '<UNK>']
                 for w in str_words]
        tag = tag_to_id[s[1]] # if s[1] gives KeyValue error, check generation of tag_to_id
        if len(words)==0:
            continue
        data.append({
            'str_words': str_words,
            'words': words,
            'tag': tag,
        })
    return data

def pad_seq(seq, max_length, PAD_token=0):
    # print(max_length) 20ng 635
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def create_batches(dataset, batch_size, order='keep'):

    newdata = copy.deepcopy(dataset)
    if order=='sort':
        newdata.sort(key = lambda x:len(x['words']))
    elif order=='random':
        random.shuffle(newdata)

    newdata = np.array(newdata)  
    batches = []
    num_batches = np.ceil(len(dataset)/float(batch_size)).astype('int')

    for i in range(num_batches):
        batch_data = newdata[(i*batch_size):min(len(dataset),(i+1)*batch_size)]

        words_seqs = [itm['words'] for itm in batch_data]
        target_seqs = [itm['tag'] for itm in batch_data]
        str_words_seqs = [itm['str_words'] for itm in batch_data]

        seq_pairs = sorted(zip(words_seqs, target_seqs, str_words_seqs, 
                               range(len(words_seqs))), key=lambda p: len(p[0]), reverse=True)

        words_seqs, target_seqs, str_words_seqs, sort_info = zip(*seq_pairs)
        
        words_lengths = np.array([len(s) for s in words_seqs])
        words_padded = np.array([pad_seq(s, np.max(words_lengths)) for s in words_seqs])
        words_mask = (words_padded!=0).astype('int')

        outputdict = {'words':words_padded, 'tags': target_seqs, 'wordslen': words_lengths,
                      'tagsmask':words_mask, 'str_words': str_words_seqs, 'sort_info': sort_info}

        batches.append(outputdict)

    return batches

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)

def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)

def bayes_loss_function(l_pw, l_qw, l_likelihood, n_batches, batch_size):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)

# def align(x, y, start_dim=2):
#     xd, yd = x.dim(), y.dim()
#     if xd > yd:
#         for i in range(xd - yd): y = y.unsqueeze(0)
#     elif yd > xd:
#         for i in range(yd - xd): x = x.unsqueeze(0)
#     xs = list(x.size())
#     ys = list(y.size())
#     nd = len(ys)
#     for i in range(start_dim, nd):
#         td = nd-i-1
#         if   ys[td]==1: ys[td] = xs[td]
#         elif xs[td]==1: xs[td] = ys[td]
#     return x.expand(*xs), y.expand(*ys)

# def dot(x, y):
#     x, y = align(x, y)
#     assert(1<y.dim()<5)
#     if y.dim() == 2:
#         return x.mm(y)
#     elif y.dim() == 3: 
#         return x.bmm(y)
#     else:
#         xs,ys = x.size(), y.size()
#         res = torch.zeros(*(xs[:-1] + (ys[-1],)))
#         for i in range(xs[0]): res[i] = x[i].bmm(y[i])
#         return res

# def aligned_op(x,y,f):
#     x, y = align(x,y,0)
#     return f(x, y)

# def add(x, y): return aligned_op(x, y, torch.add)
# def sub(x, y): return aligned_op(x, y, torch.sub)
# def mul(x, y): return aligned_op(x, y, torch.mul)
# def div(x, y): return aligned_op(x, y, torch.truediv)