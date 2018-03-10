import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import adjusted_rand_score
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *

from scipy.spatial.distance import cosine
from hierarchy import get_reference, compute_clusters, plot_distance_matrix, plot_dendrogram

# fetch data
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

nclusters = 5 # what's a cluster exaclty???

# Reference standard
print('Reference')
body_hierarchy, body_parts, reference_matrix, reference_connectvity = get_reference()
plot_distance_matrix(reference_matrix, body_parts)
cl_ref = compute_clusters(reference_matrix, body_parts, nclusters)
plot_dendrogram(cl_ref, labels=body_parts)

n = len(body_parts)
#############
# word2vec
# Compute the similarity matrix according to word2vec
print('\nword2vec')
glove_input_file = 'glove.6B.50d.txt'
word2vec_output_file = './GoogleNews-vectors-negative300.bin'#'glove.6B.100d.txt.word2vec'
#word2vec_output_file = 'glove.6B.100d.txt.word2vec'
#glove2word2vec(glove_input_file, word2vec_output_file)
model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=True) # './GoogleNews-vectors-negative300.bin'

distance_matrix_w2v = np.zeros(shape=(n,n), dtype='float64')
for i,j in np.ndindex((n,n)):
    distance_matrix_w2v[i,j] = model.distance(body_parts[i], body_parts[j])
    
plot_distance_matrix(distance_matrix_w2v, body_parts)
cl_w2v = compute_clusters(distance_matrix_w2v, body_parts, nclusters)
plot_dendrogram(cl_w2v, labels=body_parts)

print('word2vec vs REF: {}'.format(adjusted_rand_score(cl_ref.labels_, cl_w2v.labels_)))

#############
# LSA/LDA
num_features = 10000 # most frequent N words included

# Latent Semantic Analysis (LSA) traditionally uses Singular
# Value Decomposition (SVD) truncating lower dimensional components
# from the decomposition. However, a pitfalls of using SVD is that 
# the truncated matrix will have negative components, which is not 
# natural for interpreting the textual representations. 
# Nonnegative Matrix Factorization (NMF) alleviates this issue.

# Non-negative Matrix Factorization (NMF) uses TF-IDF 
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = np.array(tfidf_vectorizer.get_feature_names())

# LDA uses raw term counts because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=num_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = np.array(tf_vectorizer.get_feature_names())


def show_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print "Topic %d:" % (topic_idx)
        print " ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# Run LDA
lda = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

# retrieve body parts vectors
body_vectors_nmf = []
body_vectors_lda = []
n = len(body_parts)

# Retrieve vectors representations of body parts according to the models (i.e. the components'
# columns vectors). If not found, just set it to 0
for part in body_parts:
    part_idx = np.where(tfidf_feature_names == unicode(part))
    if len(part_idx[0]) > 0:
        v_nmf, v_lda = nmf.components_[:,part_idx].squeeze(), lda.components_[:,part_idx].squeeze()
        body_vectors_nmf.append(v_nmf)
        body_vectors_lda.append(v_lda)
    else:
        print('{} not found'.format(part))
        body_vectors_nmf.append(np.zeros((no_topics)))
        body_vectors_lda.append(np.zeros((no_topics)))
        
distance_matrix_nmf = np.zeros(shape=(n,n), dtype='float64')
distance_matrix_lda = np.zeros(shape=(n,n), dtype='float64')

for i,j in np.ndindex((n,n)):
    dist_nmf = cosine(body_vectors_nmf[i], body_vectors_nmf[j])
    dist_lda = cosine(body_vectors_lda[i], body_vectors_lda[j])
    
    # if one of the vector could not be found, set the distance to the max (1)
    distance_matrix_nmf[i,j] = 1 if np.isnan(dist_nmf) else dist_nmf
    distance_matrix_lda[i,j] = 1 if np.isnan(dist_lda) else dist_lda

print('\nNMF')
plot_distance_matrix(distance_matrix_nmf, body_parts)
cl_nmf = compute_clusters(distance_matrix_nmf, body_parts, nclusters)
plot_dendrogram(cl_nmf, labels=body_parts)

print('NMF vs. REF: {}'.format(adjusted_rand_score(cl_ref.labels_, cl_nmf.labels_)))

print('\nLDA')
plot_distance_matrix(distance_matrix_lda, body_parts)
cl_lda = compute_clusters(distance_matrix_lda, body_parts, nclusters)
plot_dendrogram(cl_lda, labels=body_parts)

print('LDA vs. REF: {}'.format(adjusted_rand_score(cl_ref.labels_, cl_lda.labels_)))
#
#
#num_top_words = 10
#
#print('Topics according to NMF')
#show_topics(nmf, tfidf_feature_names, num_top_words)
#print('Topics according to LDA')
#show_topics(lda, tf_feature_names, num_top_words)
