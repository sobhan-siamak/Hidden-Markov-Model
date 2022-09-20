


import numpy as np
import pandas as pd
import matplotlib as plt
# import nltk

# Preprocessing Data for Supervised



############ Preprocessing Data for Supervised ##############
data = pd.read_csv('hmm-train.txt', header=None)
tag = pd.read_csv('tagss.txt', sep="\t", header=None, error_bad_lines=False)
datatag = tag.iloc[:, 0]
# print(datatag)
# data1=data.iloc[:,0]
words=[]
for i in range(len(data)):
    a = data.iloc[i, 0] = data.iloc[i, 0][2:len(data.iloc[i, 0]) - 1]
    words.append(a)
# print(data)

  # words= a.copy()

print(words)
print(len(data))
tags=[]
for j in range(len(data)):
    b = data.iloc[j, 1] = data.iloc[j, 1][2:len(data.iloc[j, 1]) - 2]
    tags.append(b)
print(tags)
print(len(tags))

# wordsun=words.unique()
wordsun=list(set(words))
# print(len(wordsun))
wordsnumb=len(wordsun)####  = #words
tagsun=list(set(tags))
# print((tagsun))
tagsnumb=len(tagsun)####  = #tags

sent = ["why", "should", "you", "think", "like", "that"]
sentag = ["WRB", "MD", "PRP", "VB", "VB", "DT"]  ##### manually taged by dataset
snumb=len(sent)
tnumb=len(sentag)

state = np.zeros([tnumb,tnumb])
observe = np.zeros([tnumb,snumb])


def ForwardAlgorithm(a,b,p,V):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = p * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps

            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha

def ViterbiAlgorithm(a,b,p,V):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(p * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

    # Path Array
    S = np.zeros(T)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")





    return result











