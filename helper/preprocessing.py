import numpy as np
import math
import pandas as pd
import scipy.io as spio
import sklearn.preprocessing as skpp
import scipy.sparse.linalg as ll
from os.path import abspath, exists
import os
import time

def BuildCovarianceMatrix(y):
    m,n = y.shape
    mu = np.mean(y,axis = 1)
    xc = y - mu[:,None]
    C = np.dot(xc,xc.T)/m
    return C,xc

def EigenDecomposition(K,C):
    S,W = ll.eigs(C,k = K)
    S = S.real
    W = W.real
    W = W.T
    return S,W

start = time.time()

os.chdir('/Users/mrajagopal/Documents/git-cs6242/CSE6242')

embedding = pd.read_csv('data/cord_19_embeddings_2021-05-31.csv', header=None)
embedding.insert(0, 'ID', range(0, 0 + len(embedding)))
print(embedding.head(10))
embedding.to_csv("data/embeddings.csv", sep=",",index=False)

embedding_nocordid = embedding.loc[:, embedding.columns != 0]
A1 =embedding_nocordid.to_numpy()
A1 = A1.T

C1,xc1 = BuildCovarianceMatrix(A1)
S1,W1 = EigenDecomposition(2,C1)
principal_component1 = np.dot(W1.T[:,0],xc1)/math.sqrt(S1[0])
principal_component2 = np.dot(W1.T[:,1],xc1)/math.sqrt(S1[1])
index = np.arange(len(principal_component1))
Principal_components = np.column_stack((index,principal_component1,principal_component2))

df = pd.DataFrame(Principal_components, columns=['index','principal_component1','principal_component2'])
df.to_csv("data/principal_components.csv", sep=",",index=False)

end = time.time()
print (end-start)
