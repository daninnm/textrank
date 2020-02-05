import numpy as np
import pandas as pd
import networkx as nx
import nltk
import re
import sys

from collections import defaultdict
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# Get the name of the input file
infile = 'tests/bulgaria.txt'
if len(sys.argv) > 1:
    infile = sys.argv[1]
    print('Using filename', infile)
else:
    print('Using default filename', infile)

# Load the word embeddings (bulgarian) into memory
print('Loading embeddings')
word_embeddings = defaultdict(lambda: np.zeros(100)) # They are vectors of length 100
with open('embeddings-bg.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs

print('Clearing sentences')
text = open(infile, 'rb').read().decode('utf-8')
sents = sent_tokenize(text)
# Remove all characters that are not letters and lowercase the letters
clean_sents = [re.sub(r'([^\w]|[\d_])+', ' ', s).lower().split() for s in sents]
stop_words = 'на от за да се по са ще че не това си като до през които най при но има след който към бъде той още може му което много със която или само тази те обаче във вече около както над така между ако лв им тези преди млн бе също пред ни когато защото кв би пък тъй ги ли пак според този все някои'.split()
def remove_stopwords(sent):
    return [x for x in sent if x not in stop_words]
clean_sents = [remove_stopwords(sent) for sent in clean_sents]
# Each sentence is an array of words (without stopwords)

print('Vectorizing sentences')
# Each sentence is the mean of the vectors corresponding to its words
sent_vectors = []
for sent in clean_sents:
    if sent:
        v = sum([word_embeddings[word] for word in sent])/(len(sent)+0.001)
    else:
        v = np.zeros(100)
    sent_vectors.append(v.reshape(1, 100))

print('Creating similarity matrix')
n = len(sents)
sim_mat = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sent_vectors[i], sent_vectors[j])[0,0]

print('Applying PageRank')
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)
# Sentences on topics which are explained in detail are considered the most summerizing

ranked_sents = sorted(((scores[i], s) for i, s in enumerate(sents)), reverse=True)
for i in range(10):
    print(ranked_sents[i][1]+'\n------------------------------------------------------')
