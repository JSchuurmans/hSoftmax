import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset

train = open('data/trec/trec_train.txt', encoding='utf-8').read().split('\n')
train = open('data/trec/trec_train.txt', encoding='utf-8').read().split('\n')

en = spacy.load('en')

def tokenize(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]

TEXT = Field(tokenize=tokenize)

import pandas as import pdb; pdb.set_trace()

raw_data = {''}