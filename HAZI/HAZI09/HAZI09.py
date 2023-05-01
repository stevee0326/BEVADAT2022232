#!/usr/bin/env python
# coding: utf-8

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import sklearn.datasets as datasets

class KMeansOnDigits:
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self):
        self.digits = datasets.load_digits()

    def predict(self):
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = self.model.fit_predict(self.digits.data)

    def get_labels(self):
        labels = np.zeros_like(self.clusters)
        for i in range(self.n_clusters):
            mask = (self.clusters == i)
            labels[mask] = np.bincount(self.digits.target[mask]).argmax()
        self.labels = labels

    def calc_accuracy(self):
        self.accuracy = round(accuracy_score(self.labels, self.labels), 2)
    
    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
        #return self.mat


# In[23]:


#kmeans_model = KMeansOnDigits(n_clusters = 10, random_state = 0)
#kmeans_model.load_dataset()
#kmeans_model.predict()
#kmeans_model.get_labels()
#kmeans_model.calc_accuracy()
#kmeans_model.confusion_matrix()

