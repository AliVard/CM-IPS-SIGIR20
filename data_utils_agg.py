'''
Created on 17 Oct 2019

@author: aliv
'''

from __future__ import print_function


import numpy as np
from sklearn.datasets import load_svmlight_file

import sys
import time
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CM_IPS.mLogging import logger


class ReadRelevanceData():
  
  def __init__(self, query_file_path, click_file_path, topk):
    with open(query_file_path, 'rb') as f:
      queries = pickle.load(f)
    self.X = queries['X']
    self.y = queries['y']
    self.qid = queries['qid']
    
    logger.info('queries:{}, docs:{}'.format(self.qid.shape[0], self.y.shape[0]))
    
    self.samples_size = self.qid.shape[0]
    self.last_used_batch_index = -1
    self.embed_size = self.X.shape[1]
    self.topk = topk
    if click_file_path is not None:
      with open(click_file_path, 'rb') as f:
        self.clicks = pickle.load(f)
      
    
  def load_batch(self, indexes):
    self.x_train = np.zeros([len(indexes) * self.topk, self.embed_size], dtype=np.float32)
    self.y_train = []
    
    for i in range(len(indexes)):
    # shape of each value: (1, embed_size, topk)
      self.x_train[i*self.topk:(i+1)*self.topk, :] = self.X[indexes[i]*self.topk:(indexes[i]+1)*self.topk, :]
      self.y_train.append(self.clicks[self.qid[indexes[i]]])
    
  
  def get_random_indexes(self, batch_size):
    if self.last_used_batch_index == -1:
      self.epochs = -1
      self.last_used_batch_index = self.samples_size
    
    if self.last_used_batch_index >= self.samples_size:
      self.permuted_indices = np.random.permutation(self.samples_size)
      self.last_used_batch_index = 0
      self.epochs += 1
      
    start = self.last_used_batch_index
    end = self.last_used_batch_index + batch_size
    if end > self.samples_size:
      end = self.samples_size
    self.last_used_batch_index = end
    
    return self.permuted_indices[start:end]

def read_relevance_data(query_file_path, click_file_path, required_clicks = -1, topk=20):
  return ReadRelevanceData(query_file_path, click_file_path, topk)
