'''
GoW Kernels -- Kernel Methods for Machine Learning 2017-2018 -- RATNAMOGAN Pirashanth -- SAYEM Othmane
This file contains some implementations of functions that allows to compute the gram matrices for the following 
kernels : shortest path kernel, Weisfeiler-Lehman kernel, graphlet kernel
Some of thoses functions are taken from the MVA Altegrad class 
http://math.ens-paris-saclay.fr/version-francaise/formations/master-mva/contenus-/advanced-learning-for-text-and-graph-data-239506.kjsp?RH=1242430202531
'''
import numpy as np
import pandas as pd
import igraph
import networkx as nx
import itertools
from scipy.sparse import lil_matrix
from collections import defaultdict
import copy

ngrams = lambda a, n: list(zip(*[a[i:] for i in range(n)]))
ngrams_join =lambda data,n: [''.join(a) for a in ngrams(data,n)]


X_train_0 = (pd.read_csv(r'.data/Xtr0.csv',header=None).values).tolist()
X_train_1 = (pd.read_csv(r'.data/Xtr1.csv',header=None).values).tolist()
X_train_2 = (pd.read_csv(r'.data/Xtr2.csv',header=None).values).tolist()

    
X_train_0 = (np.array(X_train_0)[:,0]).tolist()
X_train_1 = np.array(X_train_1)[:,0].tolist()
X_train_2 = np.array(X_train_2)[:,0].tolist()


ng00 = ngrams_join(X_train_0[0],6)
ng01 = ngrams_join(X_train_0[1],6)

def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence within a sliding window of fixed size 'w'.
    '''
    
    from_to = {}
    
    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(min(w,len(terms_temp))), r=2))
    
    new_edges = []
    
    for my_tuple in indexes:
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))
    
    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i] # term to consider
        terms_temp = terms[(i-w+1):(i+1)] # all terms within sliding window
        
        # edges to try
        candidate_edges = []
        for p in range(w-1):
            candidate_edges.append((terms_temp[p],considered_term))
    
        for try_edge in candidate_edges:
            
            if try_edge[1] != try_edge[0]:
            
            # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1
                                   
                # if edge has never been seen, create it and assign it a unit weight     
                else:
                    from_to[try_edge] = 1
    
    # create empty graph
    g = igraph.Graph(directed=True)
    
    # add vertices
    g.add_vertices(sorted(set(terms)))
    
    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())
    
    # set edge and vertex weights
    g.es['weight'] = from_to.values() # based on co-occurence within sliding window
    g.vs['weight'] = g.strength() # weighted degree
    
    return(g)
    


def create_all_questions_graphs(data,w=2,n=6):
    ''' 
    Creates list of graphs for all questions
    params : w window of the bow representation 
             n n for ngrams
    return : list of graphs 
    '''
    q_graph=[]
    for q in data:
        ng = ngrams_join(q,n)
        q_igraph = terms_to_graph(ng,w)
        edges = q_igraph.get_edgelist()
        q_graph_i = nx.Graph(edges)
        nodes = q_graph_i.nodes()
        for v in nodes:
            q_graph_i.node[v]['label'] = q_igraph.vs['name'][v]
        q_graph.append(q_graph_i)
    return q_graph

    

def sp_kernel(graphs):
    """
    Computes the shortest path kernel
    and returns the kernel matrix.

    Parameters
    ----------
    graphs : list
    A list of NetworkX graphs

    Returns
    -------
    K : numpy matrix
    The kernel matrix

    """
    N = len(graphs)
    all_paths = {}
    sp_counts = {}
    for i in range(N):
        sp_lengths = nx.shortest_path_length(graphs[i]) # dictionary containing, for each node, the length of the shortest path with all other nodes in the graph
        sp_counts[i] = {}
        nodes = graphs[i].nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    label = tuple(sorted([graphs[i].node[v1]['label'], graphs[i].node[v2]['label']]) + [sp_lengths[v1][v2]])
                    # frequency of each label ('feature') in the graph
                    if label in sp_counts[i]:
                        sp_counts[i][label] += 1
                    else:
                        sp_counts[i][label] = 1
                    
                    # index of label in feature space
                    if label not in all_paths:
                        all_paths[label] = len(all_paths)

    phi = np.zeros((N,len(all_paths)))
    
    # construct feature vectors of each graph
    for i in range(N):
        for label in sp_counts[i]:
            phi[i,all_paths[label]] = sp_counts[i][label]

    K = np.dot(phi,phi.T)

    return K

def wl_kernel(graphs, h):
    """
    N.B : this kernel was taken from a code provided in the Altegrad course for MVA
    Computes the Weisfeiler-Lehman kernel by performing h iterations
    and returns the kernel matrix.
    
    Parameters
    ----------
    graphs : list
    A list of NetworkX graphs

    h : int
    The number of WL iterations

    Returns
    -------
    K : numpy matrix
    The kernel matrix

    """     
    labels = {}
    label_lookup = {}

    N = len(graphs)

    orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

    # initial labeling
    ind = 0
    for G in graphs:
        labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
        node2index = {}
        for node in G.nodes():
            node2index[node] = len(node2index)
            
        for node in G.nodes():
            label = G.node[node]['label']
            if not label in label_lookup: # has_key(label) in Python 2
                label_lookup[label] = len(label_lookup)

            labels[ind][node2index[node]] = label_lookup[label]
            orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1
        
        ind += 1
        
    compressed_labels = copy.deepcopy(labels)

    # WL iterations
    for it in range(h):
        label_lookup = {}
        ind = 0
        for G in graphs:
            node2index = {}
            for node in G.nodes():
                node2index[node] = len(node2index)
                
            for node in G.nodes():
                node_label = tuple([labels[ind][node2index[node]]])
                neighbors = G.neighbors(node)
                if len(neighbors) > 0:
                    neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
                    node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
                if not node_label in label_lookup: # has_key(node_label) in Python 2
                    label_lookup[node_label] = len(label_lookup)
                    
                compressed_labels[ind][node2index[node]] = label_lookup[node_label]
                orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1
                
            ind +=1
            
        labels = copy.deepcopy(compressed_labels)

    K = np.zeros((N, N))

    for it in range(-1, h):
        for i in range(N):
            for j in range(N):
                common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
                K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][j].get(k,0) for k in common_keys])                                    
    return K


def gr_kernel(graphs):
    """
    N.B : this kernel was taken from a code provided in the Altegrad course for MVA
    Computes the graphlet kernel for connected graphlets of size 3
    and returns the kernel matrix.

    Parameters
    ----------
    graphs : list
    A list of NetworkX graphs

    Returns
    -------
    K : numpy matrix
    The kernel matrix

    """
    labels = {}
    
    for G in graphs:
        for node in G.nodes():
            if G.node[node]["label"] not in labels:
                labels[G.node[node]["label"]] = len(labels)
    
    L = len(labels)
    B=2*pow(L,3)

    phi = lil_matrix((len(graphs),B))

    graphlets = {} # which col of phi each graphlet corresponds to
    # 'graphlets[graphlet]' will give column corresponding to feature (graphlet)
    # 'ind' gives row corresponding to graph

    ind = 0
    for G in graphs:
        for node1 in G.nodes():
            for node2 in G.neighbors(node1):
                for node3 in G.neighbors(node2):
                    if node1 != node3:
                        if node3 not in G.neighbors(node1): # open triangle
                            graphlet = (1, min(G.node[node1]['label'], G.node[node3]['label']), G.node[node2]['label'], max(G.node[node1]['label'], G.node[node3]['label']))
                            increment = 1.0/2.0 # will re-appear when node 3 is node 1
                        else: # triangle case
                            labs = sorted([G.node[node1]['label'], G.node[node2]['label'], G.node[node3]['label']])
                            graphlet = (2, labs[0], labs[1], labs[2])
                            increment = 1.0/6.0

                        if graphlet not in graphlets:
                            graphlets[graphlet] = len(graphlets)

                        phi[ind,graphlets[graphlet]] = phi[ind,graphlets[graphlet]] + increment # must be an integer in the end - in the loops, can be 0.5, 1.5, etc.
                        
        ind += 1

    K = np.dot(phi,phi.T)

    return np.asarray(K.todense())


def normalizekm(K):
    v = np.sqrt(np.diag(K));
    nm =  np.outer(v,v)
    Knm = np.power(nm, -1)
    Knm = np.nan_to_num(Knm) 
    normalized_K = K * Knm;
    return normalized_K


graphs = create_all_questions_graphs(X_train_0[:1000],w=2,n=6)
K = sp_kernel(graphs)
k = normalizekm(K)