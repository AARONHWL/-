#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import os
import math
import requests
import numpy as np

from gensim.models.word2vec import Word2Vec
from gensim.corpora.wikicorpus import WikiCorpus
from tqdm import tqdm

#使用tqdm下載原始資料
def download_file(url, file):
    r = requests.get(url, stream=True)

    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0
    with open(file, 'wb') as f:
        for data in tqdm(r.iter_content(block_size), total=math.ceil(total_size / block_size), unit='KB',unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

#看你需不需要下載資料
def download_wikidump(file):
    url = 'https://dumps.wikimedia.org/enwiki/20191120/enwiki-20191120-pages-articles-multistream1.xml-p10p30302.bz2'
    if not os.path.exists(file):
        download_file(url, file)
    else:
        logging.info('%s exists, skip download', file)

#解析維基資料
class WikiSentences:
    def __init__(self, wiki_dump_path):
        self.wiki = WikiCorpus(wiki_dump_path)
        

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            yield list(sentence)

#找相近詞彙
def find_k_nearest(source, vectors, k):
    norm1 = np.linalg.norm(source)
    norm2 = np.linalg.norm(vectors, axis=1)
    cosine_similarity = np.sum(source * vectors, axis=1) / norm1 / norm2
    return np.argsort(cosine_similarity)[::-1][1:(k + 1)]

def main():
    
    WIKIXML = 'enwiki.xml.bz2'

    #下載維基資料
    download_wikidump(WIKIXML)

    # 解析資料
    wiki_sentences = WikiSentences(WIKIXML)
    
    #建立word2vec model
    logging.info('Training model %s', 'word2vec')
    model = Word2Vec(wiki_sentences, sg=1, hs=1, size=200, workers=4, iter=5, min_count=10)
    logging.info('Training done.')
    model.save('w2v_train')
    #model = model.Word2Vec.load("w2v_train") 若需呼叫model
    
    #把詞向量儲存
    logging.info('Save trained word vectors')
    with open('en_wiki-00.txt', 'w', encoding='utf-8') as f:
        f.write('%d %d\n' % (len(model.wv.vocab),200))
        for word in tqdm(model.wv.vocab):
            f.write('%s %s\n' % (word, ' '.join([str(v) for v in model.wv[word]])))
    
    #打開詞向量的檔案，然後沿著直向堆疊向量
    words = []
    vectors = []
    logging.info('Loading word vector')
    with open('en_wiki-00.txt','r',encoding='utf-8') as f:
        f.readline()
        line = f.readline()
        while len(line) > 0:
            line = line.split(' ')
            words.append(line[0])
            vectors.append(np.array([float(x) for x in line[1:]]))
            line = f.readline()
    vectors = np.vstack(vectors)
    
    #找近似詞
    k = 5
    target_words = ['stock', 'fear', 'europe', 'slump', 'rise']
    for word in target_words:
        word_index = words.index(word)
        k_nearest = find_k_nearest(vectors[word_index], vectors, k)
        logging.info('Nearest words of %s', word)
        for index in k_nearest:
            v1 = vectors[word_index, :]
            v2 = vectors[index, :]
            logging.info('word %s score %f', words[index], np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    os.makedirs('\data',exist_ok=True)
    main()


# In[ ]:




