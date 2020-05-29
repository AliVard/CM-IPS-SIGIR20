'''
Created on 12 Sep 2019

@author: aliv
'''
import numpy as np
from math import log
import os

from absl import app
from absl import flags

from sklearn.datasets import load_svmlight_file

if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
  
  flags.DEFINE_integer( 'topk', 10, '')
  flags.DEFINE_string(  'path', 'Data/set1.train.b.lgbm_800_2.top20.txt', '')
  flags.DEFINE_bool(    'svm', True, 'is the file in the svmlight format or just a two column prediction file.')
  flags.DEFINE_list(    'eval_at', ['1', '3', '5', '10'], 'list of "k" values for evaluating NDCG@k.')

class LTRMetrics:
  def __init__(self,y,query_count, y_pred = None):
#     self._y = y
#     self._y_pred = y_pred
#     self._query_count = query_count
    
    self._querySeparatedMap = {}
    pos = 0
    for i, cnt in enumerate(query_count):
      tmp_y = np.array(y[pos:pos+cnt], copy=True)
      if not y_pred is None:
        tmp_y = tmp_y[y_pred[pos:pos+cnt].argsort()[::-1]]
        
#       jointList = np.array(list(zip(y_pred[pos:pos+cnt],y[pos:pos+cnt])))
#       self._querySeparatedMap[i] = jointList[jointList[:,0].argsort()[::-1]]
      self._querySeparatedMap[i] = np.array(tmp_y)
      pos += cnt
      
  def DCG(self, k):
    dcg = 0
    for _,docs in self._querySeparatedMap.items():
      if k > len(docs):
        k = len(docs)
      for i in range(k):
#         dcg += (2**docs[i,1]-1)/log2(i+1+1)
        dcg += (2**docs[i]-1)/log(i+1+1, 2)
        
    return dcg/len(self._querySeparatedMap)
  
  def NDCG(self, k):
#     zero_dcg = 0
    ndcg = 0
    denom = 0
    for _,docs in self._querySeparatedMap.items():
      k_ = k
      if k > len(docs):
        k_ = len(docs)
      
      dcg = 0
      idcg = 0
      for i in range(k_):
#         dcg += (2**docs[i,1]-1)/log2(i+1+1)
        dcg += (2**docs[i]-1)/log(i+1+1, 2)
        
#       if dcg == 0:
#         zero_dcg += 1
      docs_ = np.array(docs[:],copy=True)
#       docs_ = np.array(docs[:,1],copy=True)
      docs_.sort()
      docs_ = docs_[::-1]
      for i in range(k_):
        idcg += (2**docs_[i]-1)/log(i+1+1, 2)
        
      if idcg > 0:
        ndcg += dcg/idcg
        denom += 1
    
#     print('sum:{}, zeros:{}'.format(ndcg,zero_dcg))
    return ndcg/denom
  
  
  def queryCount(self):
    return len(self._querySeparatedMap)
  
def eval_output(y_true, y_pred, query_counts):
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred)
  
  return [ltr.NDCG(int(k)) for k in range(1,11)]
  
def eval_predictions(path, eval_at, query_counts=10):
  predicts = np.genfromtxt(path, delimiter=',')
  y_pred = predicts[:,0]
  y_true = predicts[:,1]
  if isinstance(query_counts, int):
    query_counts = np.ones([int(len(y_pred)/query_counts)],
                           dtype=np.int)*query_counts

  ltr = LTRMetrics(y_true,query_counts,y_pred)
  ltr_orig = LTRMetrics(y_true,query_counts)
  
  print('{} -> {}'.format(os.path.basename(path), [ltr.NDCG(int(k)) for k in eval_at]))
  print('original -> {}'.format([ltr_orig.NDCG(int(k)) for k in eval_at]))

def eval_svmlight(file_path, eval_at):
  _, y, q = load_svmlight_file(file_path, query_id=True)
  _, q = np.unique(q,return_counts=True)
  ltr = LTRMetrics(y,q)

  print('{} -> {}'.format(os.path.basename(file_path), [ltr.NDCG(int(k)) for k in eval_at]))
    
def main(argv):
  if FLAGS.svm:
    eval_svmlight(FLAGS.path, FLAGS.eval_at)
  else:
    eval_predictions(FLAGS.path, FLAGS.eval_at, FLAGS.topk)
  
if __name__ == '__main__':
  app.run(main)
