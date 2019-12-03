#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gensim
import pandas as pd

#記得要先去載google訓練好的詞庫: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

words = ['market', 'queen', 'teacher','slump']  # 建立我們要找的詞
similar_words = {}
for k in words:
    if k in model.vocab:  # 確認在資料集當中有沒有這個詞
        df[k] = [word for word, score in model.most_similar(k)]  
    else:
        print(k, ' not in vocab')
similar_words = pd.DataFrame(df)
simiar_words

