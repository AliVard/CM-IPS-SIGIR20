'''
Created on 18 Nov 2019

@author: aliv
'''

import os
import numpy as np

import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

from absl import app
from absl import flags
import pickle

FLAGS = flags.FLAGS

flags.DEFINE_string(  'file_path', 'Data/set1.test.b.txt',
                      'light svm file path to be under-sampled.')
flags.DEFINE_string(  'output_dir', None, #'Data/train',
                      'output directory. Leave it "None" for a same directory as the "file_path".')
flags.DEFINE_string(  'model_path', 'lambdaMART/lgbm_50_2.txt', 
                      'LambdaMART model path as the base ranker.')

flags.DEFINE_integer( 'k', 20,
                      'topk')
                      

# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Topk:
  def __init__(self, input_path, model_file):
    X, y, q = load_svmlight_file(input_path, query_id=True)
    qid, qcnt = np.unique(q,return_counts=True)
    
    print('input file svm data loaded!')
    
    self._booster = lgb.Booster(model_file=model_file)
    
    self._y_pred = self._booster.predict(X)
    self._qid = qid
    self._qcnt = qcnt
    self._y = y
    self._X = X
    
    self._output_name = os.path.splitext(os.path.basename(input_path))[0] + '.' + os.path.splitext(os.path.basename(model_file))[0]
    
  def rankAndCut(self, output_dir, k):
    new_X = np.zeros([self._qid.shape[0] * k, self._X.shape[1]], dtype=np.float32)
    new_y = np.zeros([new_X.shape[0]], dtype=np.float32)
    self._ranked_lines = []
    pos = 0
    for i, l in enumerate(self._qcnt):
#       if l < 2:
#         pos += l
#         continue
      y_this = np.array(self._y_pred[pos:pos+l],copy=True)
      args = y_this.argsort()
      args = args[::-1]
      args = args[:min(k,l)]
      
#       print(args)
      new_X[i*k:(i*k+args.shape[0]),:] = self._X[pos+args,:].todense()
      new_y[i*k:i*k+args.shape[0]] = self._y[pos+args]
        
      pos += l
      
    ranked = {'X':new_X, 'y':new_y, 'qid':self._qid}
    output_file = self._output_name + '.top{}.pkl'.format(k)
    with open(os.path.join(output_dir, output_file), 'wb') as f:
      pickle.dump(ranked, f, 2)
    print('saved {} successfully!'.format(output_file))
        
def main(args):
  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_dir = os.path.dirname(os.path.abspath(FLAGS.file_path))
    
  topk = Topk(FLAGS.file_path, FLAGS.model_path)
  topk.rankAndCut(output_dir, FLAGS.k)
  
      
  
if __name__ == '__main__':
  app.run(main)