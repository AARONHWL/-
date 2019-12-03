#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter


def preprocess(text):
    stop = set(stopwords.words("english"))
    text = re.sub("[0-9\!\%\[\]\,\.\‘\’\|\-;\:\(\)\&\$\—\?\/]","",text)
    text = text.lower()
    token_words = word_tokenize(text)  
    words =[word for word in token_words if word not in stop]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t, pos='v') for t in words]
    return text

def cal_freq(words,freq):
    vocab = set(words)
    freq_str = str(freq)
    vocab_to_int = {w: c for c, w in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    int_words = [vocab_to_int[w] for w in words]
    int_word_counts = Counter(int_words)
    total_count = len(int_words)
    word_freqs = {w: c/total_count for w, c in int_word_counts.items()}

    a = pd.Series(int_to_vocab)
    b =pd.Series(word_freqs)
    freq_table = pd.DataFrame(a,columns=['words'])
    freq_table.insert(1,freq_str,b)
    return freq_table


for i in range(1,156):
    with open('CNBC\\%s.txt'%i,'r', encoding='utf-8') as f:
        locals()['texts%s'%i] = f.read()
    locals()['words%s'%i] = preprocess(locals()['texts%s'%i])
    locals()['table%s'%i]= cal_freq(locals()['words%s'%i],'freq%s'%i)
merge_freq = pd.merge(table1,table2)
for i in range(3,156):
    merge_freq = pd.merge(merge_freq,locals()['table%s'%i],how="outer")
    merge_freq.fillna(0,inplace =True)
merge_freq
merge_freq.to_csv('CNBC_freq_table.csv')

