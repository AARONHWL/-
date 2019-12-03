#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


def find_k_nearest(source, vectors, k):
    norm1 = np.linalg.norm(source)
    norm2 = np.linalg.norm(vectors, axis=1)
    cosine_similarity = np.sum(source * vectors, axis=1) / norm1 / norm2
    return np.argsort(cosine_similarity)[::-1][1:(k + 1)]


words = []
vectors = []
with open('en_wiki-00.txt','r',encoding='utf-8') as f:
    f.readline()
    line = f.readline()
    while len(line) > 0:
        line = line.split(' ')
        words.append(line[0])
        vectors.append(np.array([float(x) for x in line[1:]]))
        line = f.readline()
#計算向量
vectors = np.vstack(vectors)

#找近似詞並建立成表格
k = 10
target_words = ['stock','slump','shock','rise']
table =pd.DataFrame([0,1,2,3,4,5,6,7,8,9], columns=['a'])
for i in range(len(target_words)):
    locals()["%s"%target_words[i]]=[]
    word_index = words.index(target_words[i])
    k_nearest = find_k_nearest(vectors[word_index], vectors, k)
    for index in k_nearest:
        nearest =words[index]
        locals()["%s"%target_words[i]].append(nearest)
    table.insert(i,target_words[i],locals()["%s"%target_words[i]])      
table.drop('a',axis=1)
#table.to_csv('similar_words')

