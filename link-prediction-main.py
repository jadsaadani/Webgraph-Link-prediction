# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import networkx as nx
import csv
import numpy as np
import community
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# First read and create the directed graph using the training dataset
G = nx.DiGraph()
G.add_nodes_from([str(k) for k in range(33226)])
with open("training.txt", "r") as f:
    for line in f:
        line = line.split()
        if line[2] == '1':
            G.add_edge(line[0], line[1])
 
# Create an undirected version of the graph           
undirG = nx.Graph()
undirG.add_nodes_from([str(k) for k in range(33226)])
with open("training.txt", "r") as f:
    for line in f:
        line = line.split()
        if line[2] == '1':
            undirG.add_edge(line[0], line[1])
            

A = nx.adjacency_matrix(G)   #adjacency matrix of G and its powers
A2 = np.dot(A,A)
A3 = np.dot(A2,A)
A4 = np.dot(A3,A)
            
            
def adamic_adar_crossed(g,i,j):
    ind = 0.
    for w in set(g.successors(i)).intersection(g.predecessors(j)):
        if len(list(g.successors(w))) > 1:
            ind += 1./np.log(len(list(g.successors(w))))
    return ind

def adamic_adar_in(g,i,j):
    ind = 0.
    for w in set(g.predecessors(i)).intersection(g.predecessors(j)):
        if len(list(g.predecessors(w))) > 1:
            ind += 1./np.log(len(list(g.predecessors(w))))
    return ind

def adamic_adar_out(g,i,j):
    ind = 0.
    for w in set(g.successors(i)).intersection(g.successors(j)):
        if len(list(g.successors(w))) > 1:
            ind += 1./np.log(len(list(g.successors(w))))
    return ind

def ressource_allocation_crossed(g,i,j):
    ind = 0.
    for w in set(g.successors(i)).intersection(g.predecessors(j)):
        if len(list(g.successors(w))) > 0:
            ind += 1./len(list(g.successors(w)))
    return ind

def ressource_allocation_in(g,i,j):
    ind = 0.
    for w in set(g.predecessors(i)).intersection(g.predecessors(j)):
        if len(list(g.predecessors(w))) > 0:
            ind += 1./len(list(g.predecessors(w)))
    return ind

def ressource_allocation_out(g,i,j):
    ind = 0.
    for w in set(g.successors(i)).intersection(g.successors(j)):
        if len(list(g.successors(w))) > 0:
            ind += 1./len(list(g.successors(w)))
    return ind



#communities detection using Louvain method
communities = community.best_partition(undirG) 

#compute TF-IDF pairwise similarities
documents = [open("node_information/text/"+str(k)+"b.txt", errors='ignore').read() for k in range(33226)]
tfidf = TfidfVectorizer().fit_transform(documents)
pairwise_similarity = tfidf * tfidf.T


X_train = np.zeros((453797,26))
X_test = np.zeros((113450,26))
Y_train = np.array([0]*453797)

#computing training features matrix :
k = 0
with open("training.txt", "r") as f:
    for line in f:
        line = line.split()
        
        Y_train[k] = int(line[2])
        
        if Y_train[k] == 1:
            G.remove_edge(line[0], line[1])

        X_train[k,0] = len(set(G.successors(line[0])).intersection(G.successors(line[1])))  #common neighbors
        X_train[k,1] = len(set(G.predecessors(line[0])).intersection(G.predecessors(line[1])))
        X_train[k,2] = len(set(G.successors(line[0])).intersection(G.predecessors(line[1])))
        
        if len(set(G.successors(line[0])).union(G.successors(line[1]))) > 0:
            X_train[k,3] = 1.*len(set(G.successors(line[0])).intersection(G.successors(line[1])))/len(set(G.successors(line[0])).union(G.successors(line[1])))  #Jaccard
        if len(set(G.predecessors(line[0])).union(G.predecessors(line[1]))) > 0:
            X_train[k,4] = 1.*len(set(G.predecessors(line[0])).intersection(G.predecessors(line[1])))/len(set(G.predecessors(line[0])).union(G.predecessors(line[1])))
       
        X_train[k,5] = adamic_adar_crossed(G,line[0],line[1])            #adamic adar
        X_train[k,6] = adamic_adar_in(G,line[0],line[1])
        X_train[k,7] = adamic_adar_out(G,line[0],line[1])
        X_train[k,8] = ressource_allocation_crossed(G,line[0],line[1])   #ressource allocation
        X_train[k,9] = ressource_allocation_in(G,line[0],line[1])
        X_train[k,10] = ressource_allocation_out(G,line[0],line[1])
        
        X_train[k,11] = len(list(G.successors(line[0])))*len(list(G.successors(line[1])))  #preferential attachment
        X_train[k,12] = len(list(G.predecessors(line[0])))*len(list(G.predecessors(line[1])))
        
        if X_train[k,11] != 0:
            X_train[k,13] = X_train[k,0]/np.sqrt(X_train[k,11])  #Neighborhood Distance (ND)
        if X_train[k,12] != 0:
            X_train[k,14] = X_train[k,1]/np.sqrt(X_train[k,12])
            
        X_train[k,15] = len(set(G.successors(line[0])).union(G.successors(line[1])))   #Total Neighbors (TD)
        X_train[k,16] = len(set(G.predecessors(line[0])).union(G.predecessors(line[1])))
        X_train[k,17] = len(set(G.successors(line[0])).union(G.predecessors(line[1])))
        
        X_train[k,18] = len(list(G.successors(line[0])))  #Node Degree
        X_train[k,19] = len(list(G.predecessors(line[0])))
        X_train[k,20] = len(list(G.successors(line[1])))
        X_train[k,21] = len(list(G.predecessors(line[1])))
        
        X_train[k,22] = pairwise_similarity[int(line[0]),int(line[1])]   #text similarity
        
        try:    #shortest length path
            X_train[k,23] = nx.shortest_path_length(G, source=line[0], target=line[1])
        except:
            X_train[k,23] = 40000    #default to large value
            
        if communities[line[0]] == communities[line[1]]:
            X_train[k,24] = 1
        else:
            X_train[k,24] = 0
            
        X_train[k,25] = A4[int(line[0]),int(line[1])]
        
        
        if Y_train[k] == 1:
            G.add_edge(line[0], line[1])
        
        k += 1
        
#computing testing features matrix :       
k = 0
with open("testing.txt", "r") as f:
    for line in f:
        line = line.split()

        X_test[k,0] = len(set(G.successors(line[0])).intersection(G.successors(line[1])))  #common neighbors
        X_test[k,1] = len(set(G.predecessors(line[0])).intersection(G.predecessors(line[1])))
        X_test[k,2] = len(set(G.successors(line[0])).intersection(G.predecessors(line[1])))
        
        if len(set(G.successors(line[0])).union(G.successors(line[1]))) > 0:
            X_test[k,3] = 1.*len(set(G.successors(line[0])).intersection(G.successors(line[1])))/len(set(G.successors(line[0])).union(G.successors(line[1])))  #Jaccard
        if len(set(G.predecessors(line[0])).union(G.predecessors(line[1]))) > 0:
            X_test[k,4] = 1.*len(set(G.predecessors(line[0])).intersection(G.predecessors(line[1])))/len(set(G.predecessors(line[0])).union(G.predecessors(line[1])))
       
        X_test[k,5] = adamic_adar_crossed(G,line[0],line[1])            #adamic adar
        X_test[k,6] = adamic_adar_in(G,line[0],line[1])
        X_test[k,7] = adamic_adar_out(G,line[0],line[1])
        X_test[k,8] = ressource_allocation_crossed(G,line[0],line[1])   #ressource allocation
        X_test[k,9] = ressource_allocation_in(G,line[0],line[1])
        X_test[k,10] = ressource_allocation_out(G,line[0],line[1])
        
        X_test[k,11] = len(list(G.successors(line[0])))*len(list(G.successors(line[1])))  #preferential attachment
        X_test[k,12] = len(list(G.predecessors(line[0])))*len(list(G.predecessors(line[1])))
        
        if X_test[k,11] != 0:
            X_test[k,13] = X_test[k,0]/np.sqrt(X_test[k,11])  #Neighborhood Distance (ND)
        if X_test[k,12] != 0:
            X_test[k,14] = X_test[k,1]/np.sqrt(X_test[k,12])
            
        X_test[k,15] = len(set(G.successors(line[0])).union(G.successors(line[1])))   #Total Neighbors (TD)
        X_test[k,16] = len(set(G.predecessors(line[0])).union(G.predecessors(line[1])))
        X_test[k,17] = len(set(G.successors(line[0])).union(G.predecessors(line[1])))
        
        X_test[k,18] = len(list(G.successors(line[0])))  #Node Degree
        X_test[k,19] = len(list(G.predecessors(line[0])))
        X_test[k,20] = len(list(G.successors(line[1])))
        X_test[k,21] = len(list(G.predecessors(line[1])))
        
        X_test[k,22] = pairwise_similarity[int(line[0]),int(line[1])]   #text similarity
        
        try:    #shortest length path
            X_test[k,23] = nx.shortest_path_length(G, source=line[0], target=line[1])
        except:
            X_test[k,23] = 40000    #default to large value
            
        if communities[line[0]] == communities[line[1]]:
            X_test[k,24] = 1
        else:
            X_test[k,24] = 0
            
        X_test[k,25] = A4[int(line[0]),int(line[1])]
        
        k += 1




model = xgb.XGBClassifier(max_depth=5, learning_rate=0.2, n_estimators=200, gamma=0, 
                           min_child_weight=2, max_delta_step=0, subsample=1, 
                           colsample_bytree=0.7, colsample_bylevel=1, colsample_bynode=1, 
                           reg_alpha=1, reg_lambda=1)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
predictions = zip(range(len(y_pred)), y_pred)
# Write the output in the format required by Kaggle
with open("finalsubmission.csv","w") as pred:
    csv_out = csv.writer(pred)
    csv_out.writerow(['id','predicted'])
    for row in predictions:
        csv_out.writerow(row) 