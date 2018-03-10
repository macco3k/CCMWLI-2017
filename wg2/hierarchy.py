from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from treelib import Node, Tree

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def create_hierarchy():
    """
    Creates a tree structure representing the body hierarchy
    """
    body_hierarchy = Tree()
    body_hierarchy.create_node('Body', 'body')
    # 1st level
    body_hierarchy.create_node('Head', 'head', parent='body')
    body_hierarchy.create_node('Trunk', 'trunk', parent='body')
    body_hierarchy.create_node('Arm', 'arm', parent='body')
    body_hierarchy.create_node('Leg', 'leg', parent='body')
    
    # 2nd level
    # head
    body_hierarchy.create_node('Face', 'face', parent='head')
    # arm
    body_hierarchy.create_node('Bicep', 'bicep', parent='arm')
    body_hierarchy.create_node('Forearm', 'forearm', parent='arm')
    body_hierarchy.create_node('Hand', 'hand', parent='arm')
    #leg
    body_hierarchy.create_node('Thigh', 'thigh', parent='leg')
    body_hierarchy.create_node('Calf', 'calf', parent='leg')
    body_hierarchy.create_node('Foot', 'foot', parent='leg')
    
    # 3rd level
    # face
    body_hierarchy.create_node('Mouth', 'mouth', parent='face')
    body_hierarchy.create_node('Nose', 'nose', parent='face')
    body_hierarchy.create_node('Lips', 'lips', parent='face')
    # hand
    body_hierarchy.create_node('Fingers', 'fingers', parent='hand')
    body_hierarchy.create_node('Palm', 'palm', parent='hand')
    
    # 4th level
    # mouth
    body_hierarchy.create_node('Teeth', 'teeth', parent='mouth')
    body_hierarchy.create_node('Tongue', 'tongue', parent='mouth')
    # fingers
    body_hierarchy.create_node('Pinky', 'pinky', parent='fingers')
    body_hierarchy.create_node('Thumb', 'thumb', parent='fingers')
    
    body_hierarchy.show()
    
    return body_hierarchy

def create_fuller_hierarchy():
    """
    Creates a tree structure representing the body hierarchy
    """
    body_hierarchy = Tree()
    body_hierarchy.create_node('Body', 'body')
    # 1st level
    body_hierarchy.create_node('Head', 'head', parent='body')
    body_hierarchy.create_node('Trunk', 'trunk', parent='body')
    body_hierarchy.create_node('Back', 'back', parent='body')
    body_hierarchy.create_node('Arm', 'arm', parent='body')
    body_hierarchy.create_node('Leg', 'leg', parent='body')
    body_hierarchy.create_node('Shoulders', 'shoulders', parent='body')
    body_hierarchy.create_node('Belly', 'belly', parent='body')
    
    # 2nd level
    # head
    body_hierarchy.create_node('Neck', 'neck', parent='head')
    body_hierarchy.create_node('Face', 'face', parent='head')
    # arm
    body_hierarchy.create_node('Bicep', 'bicep', parent='arm')
    body_hierarchy.create_node('Forearm', 'forearm', parent='arm')
    body_hierarchy.create_node('Hand', 'hand', parent='arm')
    body_hierarchy.create_node('Wrist', 'wrist', parent='arm')
    #leg
    body_hierarchy.create_node('Thigh', 'thigh', parent='leg')
    body_hierarchy.create_node('Calf', 'calf', parent='leg')
    body_hierarchy.create_node('Foot', 'foot', parent='leg')
    
    # 3rd level
    #foot
    body_hierarchy.create_node('Toes', 'toes', parent='foot')
    # face
    body_hierarchy.create_node('Cheeks', 'cheeks', parent='face')
    body_hierarchy.create_node('Chin', 'chin', parent='face')
    body_hierarchy.create_node('Forehead', 'forehead', parent='face')
    body_hierarchy.create_node('Ears', 'ears', parent='face')
    body_hierarchy.create_node('Eyes', 'eyes', parent='face')
    body_hierarchy.create_node('Eyebrows', 'eyebrows', parent='face')
    body_hierarchy.create_node('Mouth', 'mouth', parent='face')
    body_hierarchy.create_node('Nose', 'nose', parent='face')
    body_hierarchy.create_node('Lips', 'lips', parent='face')
    # hand
    body_hierarchy.create_node('Fingers', 'fingers', parent='hand')
    body_hierarchy.create_node('Palm', 'palm', parent='hand')
    
    # 4th level
    # mouth
    body_hierarchy.create_node('Teeth', 'teeth', parent='mouth')
    body_hierarchy.create_node('Tongue', 'tongue', parent='mouth')
    # fingers
    body_hierarchy.create_node('Pinky', 'pinky', parent='fingers')
    body_hierarchy.create_node('Thumb', 'thumb', parent='fingers')
    
    body_hierarchy.show()
    
    return body_hierarchy

def compute_paths_distance(tree, sort=True):
    """
    Computes the set of all distances between nodes, based on the path's length
    for each pair in the undirected graph equivalent to the tree. Distance is dependent 
    on the depth of the nodes,so that more specific (i.e. deeper) nodes are more similar 
    then nodes up the hierarchy. For example, d(pinky,fingers) < d(hand, fingers), 
    despite both having paths of length 1. Children of the same parent are assigned
    a distance of 0 in order to make them strongly tied.
    """
    # Create the graph for the tree in order to compute all paths' length
    graph = nx.Graph()
    graph.add_nodes_from(tree.nodes.keys())
    
    for p in tree.paths_to_leaves():
        graph.add_path(p)
    
    paths = nx.all_pairs_shortest_path(graph)
    distance = [(n1,n2,path,len(path)-1) for n1,nodes in paths.iteritems() 
                                         for n2,path  in nodes.iteritems()]
    
    # This will add to the bare distance, so that paths traveling up the hierarchy
    # will get a penalty
    depth = tree.depth()
    level_malus = [100*(depth-lvl)/depth for lvl in range(depth+1)]
    
    for i,(n1,n2,path,dist) in enumerate(distance):
        # Paths leading to upper level in the hierarchy should get penalized more
        # This captures far away nodes (e.g. in different subtrees), and also
        # pairs of nodes having paths of the same length, but at different depth
        # within the hierarchy
        highest_level = min(tree.level(n) for n in path)
        
        # Set brothers and sisters at no distance
        if tree.parent(n1) == tree.parent(n2):
#            print('Reassigning siblings: {},{} = 0'.format(n1, n2))
            dist = 1
                
        # Rescale distance according to depth
        if dist > 0:
            scaled_dist = level_malus[highest_level] + dist*10 # makes distances a bit larger
#            print('Rescaling {},{} from {} to {}'.format(n1, n2, dist, scaled_dist))
            
            distance[i] = (n1,n2,path,scaled_dist)
    
    if sort:
        return sorted(distance), nx.adjacency_matrix(graph)
    
    return distance, nx.adjacency_matrix(graph)

def compute_distance_matrix(distance, n):
    # loop through the list, just pick the distance
    distance_matrix = np.zeros(shape=(n,n), dtype=float)
    for (i,j),_ in np.ndenumerate(distance_matrix):
        distance_matrix[i,j] = distance[i*n + j][3]
        
    return distance_matrix / np.max(distance_matrix)

def compute_clusters(distance_matrix, names, n_clusters=2, connectivity_matrix=None, linkage='complete'):
    cl = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage=linkage, connectivity=connectivity_matrix)
    cl.fit(distance_matrix)  
    return cl

def plot_distance_matrix(distance_matrix, labels):
    n = distance_matrix.shape[0]   
    plt.figure(figsize=(20,10))
#    plt.subplot(figsize=(12,8))
    plt.imshow(distance_matrix, cmap='Blues')
    plt.xticks(range(n), labels, rotation=90)
    plt.yticks(range(n), labels)
    plt.colorbar()
    plt.show()
    
def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_

    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])

    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,5))
    dendrogram(linkage_matrix, **kwargs)
    plt.show()

def get_reference():
    body_hierarchy = create_hierarchy()
    names = sorted(body_hierarchy.nodes.keys())
    distances, connectivity_matrix = compute_paths_distance(body_hierarchy)
    distance_matrix = compute_distance_matrix(distances,len(names))
    
    return body_hierarchy, names, distance_matrix, connectivity_matrix