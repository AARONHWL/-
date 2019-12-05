#!/usr/bin/env python
# coding: utf-8

# In[16]:


import re
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd



def preprocess(text):
    stop = set(stopwords.words("english"))
    text = re.sub("[0-9\!\%\[\]\,\.\‘\’\|\-;\:\(\)\&\$\—\?\/]","",text)
    text = text.lower()
    token_words = word_tokenize(text) 
    words =[word for word in token_words if word not in stop]
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(t, pos='v') for t in words]
    return text

def ngram_table(o):
    for i in range(2,157):
        with open('CNBC\\%s.txt'%i,'r', encoding='utf-8') as f:
            text = f.read()
        tokens = preprocess(text)
        bgs = list(ngrams(tokens, o))
        fdist = nltk.FreqDist(bgs)

        words=[]
        freq=[]
        for a,b in fdist.items():
            words.append(a)
            freq.append(b)

        globals()['table%s'%i] =pd.DataFrame(list(zip(words,freq)),columns=['words','freq%s'%i])
    merge_freq = pd.merge(table2,table3)
    for k in range(4,157):
        merge_freq = pd.merge(merge_freq,globals()['table%s'%k],how="outer")
        merge_freq.fillna(0,inplace =True)
    return merge_freq

unigram_table= ngram_table(1)
bigram_table= ngram_table(2)

Table =unigram_table.append(bigram_table)
Table.set_index('words', inplace=True)
Table['col_sum'] = Table.apply(lambda x: x.sum(), axis=1)
Table.loc['row_sum'] = Table.apply(lambda x: x.sum())
Table.drop(Table[Table.col_sum < 10].index, inplace=True)
Table.to_csv('ngram_table.csv')

