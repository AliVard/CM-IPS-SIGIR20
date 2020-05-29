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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CM_IPS.mLogging import logger


class ReadRelevanceData():
  
  def __init__(self, query_file_path, click_file_path, required_clicks, topk):
      
    X, y, q = load_svmlight_file(query_file_path, query_id=True)
    qid, qcnt = np.unique(q,return_counts=True)
    
    
    
    self.embed_size = X.shape[1]
    self.topk = topk
    
    # we have to build a dictionary here. because we need "query-id"s. a list won't do!
    # shape of each value: (1, embed_size, topk)
    self.queries = {}
    if click_file_path is not None:
      pos = 0
      for i in range(len(qid)):
        self.queries[qid[i]] = np.expand_dims(X[pos:pos+self.topk,:].transpose().todense(), axis = 0)
        pos += qcnt[i]
        
      logger.info('finished reading queries!')
    
    
      self.has_clicks = True
      self._x_train_qids = np.zeros([required_clicks], dtype=np.int)
      self._y_train_all = np.zeros([required_clicks, self.topk], dtype=np.float32)
        
      click_file = open(click_file_path, 'r')
      previous_qid = -1
      read_clicks = 0
      read_sessions = -1
        
      progress_percent = required_clicks / 10
        
      start_time = time.time()
      times = []
      for line in click_file:
        read_clicks += 1
        tokens = line.split(' ')
        current_qid = int(tokens[0][4:])
        if current_qid != previous_qid:
          read_sessions += 1
          if read_clicks >= required_clicks:
            break
          self._x_train_qids[read_sessions] = current_qid
            
        if read_clicks % progress_percent == 0:
          times.append(time.time() - start_time)
          start_time = time.time()
#           print('%d.' % (read_clicks * 100 / required_clicks), end='')
        clicked_pos = int(tokens[1])
        if clicked_pos < self.topk:
          self._y_train_all[read_sessions,clicked_pos] = 1
            
        previous_qid = current_qid
          
      
      click_file.close()
      logger.info('\n reading {} clicks from {} sessions took {}'.format(read_clicks, read_sessions, sum(times)))
      
      self._x_train_qids = self._x_train_qids[:read_sessions]
      self._y_train_all = self._y_train_all[:read_sessions,:]
        
#       for i in range(10):
#         print(qid[i])
#         print(self._y_train_all[self._x_train_qids==qid[i],:])
      self.samples_size = self._x_train_qids.shape[0]
      self.last_used_batch_index = -1
    else:
      self.samples_size = int(X.shape[0]/self.topk)
      self.x_train = np.zeros([self.samples_size, self.embed_size, self.topk], dtype=np.float32)
      pos = 0
      for i in range(len(qid)):
        self.x_train[i,:,:] = np.expand_dims(X[pos:pos+self.topk,:].transpose().todense(), axis = 0)
        pos += qcnt[i]
        
#       bin_y = np.zeros_like(y, dtype=np.float)
#       bin_y[y>2.] = 1.
      self.y_train = y.reshape([-1, self.topk])
      
      logger.info('finished reading queries!')

    if click_file_path is None:
      self.x_train = self.x_train.transpose([0,2,1]).reshape([-1,self.embed_size])
#       self.y_train = self.y_train.reshape([-1, 10])

  def load_batch(self, indexes):
    self.x_train = np.zeros([len(indexes) * self.topk, self.embed_size], dtype=np.float32)
    self.y_train = np.zeros([len(indexes), self.topk], dtype=np.float32)
    
    for i in range(len(indexes)):
    # shape of each value: (1, embed_size, topk)
      self.x_train[i*self.topk:(i+1)*self.topk,:] = self.queries[self._x_train_qids[indexes[i]]][0, :, :].transpose()
      self.y_train[i,:] = self._y_train_all[indexes[i], :]
    
  
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

def read_relevance_data(query_file_path, click_file_path, required_clicks = -1, topk=10):
  return ReadRelevanceData(query_file_path, click_file_path, required_clicks, topk)
