from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering
#from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

import numpy as np
import matplotlib.pyplot as plt


model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 

# fetch data
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

# True hierarchy
body_hierarchy = {"body": { "head": {}, "trunk": {}, "arm": {}, "leg": {}}}
body_hierarchy["body"]["head"] = {"face": {}, "nose": {}, "lips": {}, "mouth": {"teeth": {}, "tongue": {}}}
#body_hierarchy["torso"] = {}
body_hierarchy["body"]["trunk"] = {}
body_hierarchy["body"]["arm"] = {"bicep": {}, "elbow": {}, "forearm": {}, "hand": 
                                                                    {"palm": {}, "fingers": 
                                                                      {"pinky": {}, "thumb": {}}
                                                                    }}
body_hierarchy["body"]["leg"] = {"thigh": {}, "calf": {}, "knee": {}, "ankle": {}, "foot": {}}

# flatten the hierarchy to extract similarities
def flatten_dictionary(dictionary):
    flattened = []
    for k,v in dictionary.items():
        if not v:
            flattened.append(k)
        else:
            flattened.extend(flatten_dictionary(v))    
            
    return flattened

n = 25
def compute_distance(hierarchy, offset=0):
    similarity = []
    # Siblings have the same similarity to each other, but this gets added one 
    # per each level they are deep (e.g. level 1 gets no bonus, level 2 get +1 etc.), 
    # so that the deeper the siblings, the more similar.
    
    # Difference between each level's siblings must be 2 (e.g. 11, 9, 7, etc.), 
    # to account for similarity between parents and children (so children are 
    
    # From leaves, to compute the similarity to the rest of the nodes, traverse
    # the tree and subtract the steps to get to the destination (intuitively, the
    # more we move away from relatives by going up the tree and down again, the
    # less similar we are to those nodes
    
    # Now, how do we implement it :D
body_parts = flatten_dictionary(body_hierarchy)

# Compute the similarity matrix according to word2vec
similarity_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
for (i,j),_ in np.ndenumerate(similarity_matrix):
    similarity_matrix[i,j] = model.similarity(body_parts[i], body_parts[j])
    
plt.imshow(similarity_matrix, cmap='Blues')
plt.xticks(range(len(body_parts)), body_parts, rotation=90)
plt.yticks(range(len(body_parts)), body_parts)
plt.colorbar()
plt.show()

# Fit using cosine similarity
cl_avg = AgglomerativeClustering(n_clusters=4, linkage='average', affinity='precomputed') # head, trunk, arm, leg
cl_avg.fit(similarity_matrix)
print('Clusters: {}'.format(cl_avg.labels_))

distance_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
for (i,j),_ in np.ndenumerate(distance_matrix):
    distance_matrix[i,j] = model.distance(body_parts[i], body_parts[j])

# Fit using distance
cl_avg.fit(distance_matrix)
print('Clusters: {}'.format(cl_avg.labels_))

#### LDA/LSA ###
    
# What are documents to compare? Possibly different hierarchies? How to extract those?

#documents = documents[documents in body_parts] ???

#num_features = 2000 # most frequent N words included

# Latent Semantic Analysis (LSA) traditionally uses Singular
# Value Decomposition (SVD) truncating lower dimensional components
# from the decomposition. However, a pitfalls of using SVD is that 
# the truncated matrix will have negative components, which is not 
# natural for interpreting the textual representations. 
# Nonnegative Matrix Factorization (NMF) alleviates this issue.

# Non-negative Matrix Factorization (NMF) uses TF-IDF 
#tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
#tfidf = tfidf_vectorizer.fit_transform(documents)
#tfidf_feature_names = tfidf_vectorizer.get_feature_names()
#
## LDA uses raw term counts because it is a probabilistic graphical model
#tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
#tf = tf_vectorizer.fit_transform(documents)
#tf_feature_names = tf_vectorizer.get_feature_names()
#
#
#def show_topics(model, feature_names, no_top_words):
#    for topic_idx, topic in enumerate(model.components_):
#        print "Topic %d:" % (topic_idx)
#        print " ".join([feature_names[i]
#                        for i in topic.argsort()[:-no_top_words - 1:-1]])
#
#no_topics = 20
#
## Run NMF
#print('Running NMF...')
#nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
#
## Run LDA
#print('Running LDA...')
#lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
#
#
#num_top_words = 10
#
#print('Topics according to NMF')
#show_topics(nmf, tfidf_feature_names, num_top_words)
#print('Topics according to LDA')
#show_topics(lda, tf_feature_names, num_top_words)
