from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from gensim.models.keyedvectors import KeyedVectors

import numpy as np
import matplotlib.pyplot as plt

from LSA_and_LDA_hierarchy import get_reference_hierarchy, compute_clusters, plot_distance_matrix

def get_w2v_cluster(model, names):
    # Compute the similarity matrix according to word2vec
    distance_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = model.distance(body_parts[i], body_parts[j])
        
    plot_distance_matrix(distance_matrix, body_parts)
    cl_w2v = compute_clusters(distance_matrix, body_parts, 9)
    
    return cl_w2v

def get_lsa_cluster(distance_matrix, names):
    # Compute the similarity matrix according to word2vec
    distance_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = model.distance(body_parts[i], body_parts[j])
        
    plot_distance_matrix(distance_matrix, body_parts)
    cl_w2v = compute_clusters(distance_matrix, body_parts, 9)
    
    return cl_w2v

def get_lda_cluster(distance_matrix, names):
    # Compute the similarity matrix according to word2vec
    distance_matrix = np.zeros(shape=(len(body_parts), len(body_parts)), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = model.distance(body_parts[i], body_parts[j])
        
    plot_distance_matrix(distance_matrix, body_parts)
    cl_w2v = compute_clusters(distance_matrix, body_parts, 9)
    
    return cl_w2v

model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 

hierarchy, body_parts, cl_ref = get_reference_hierarchy()
cl_w2v = get_w2v_cluster(model, body_parts)
print('Reference:\n{}\n'.format(cl_ref))
print('word2vec:\n{}\n'.format(cl_w2v))