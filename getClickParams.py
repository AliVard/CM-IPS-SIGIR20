'''
Created on 9 Jan 2020

@author: aliv
'''
import numpy as np
from sklearn.datasets import load_svmlight_file

import sys
import time
import os

from absl import app
from absl import flags
import pickle

FLAGS = flags.FLAGS

# flags.DEFINE_string(  'model', 'trustPBM', 'either PBM or trustPBM')
if __name__ == '__main__':
  flags.DEFINE_string(  'train_dir', '/Users/aliv/MySpace/_DataSets/LTR/Yahoo/Challenge/ltrc_yahoo/ltrc_yahoo_small/Data/', 
                                                            'directory of train files')
  flags.DEFINE_string(  'train_query_file', 'set1.train.b.lgbm_50_2.top20.pkl', 
                                                            'address of top-k document embeddings retrieved for train queries.')
  flags.DEFINE_string(  'train_click_file', 'pbm_1.0.set1.train.b.lgbm_50_2.top20.pkl', 
                                                            'address of train click logs.')
  flags.DEFINE_integer( 'topk', 20, 
                                                            'topk documents shown to user to obtain users\' feedback.')
  flags.DEFINE_string(  'output_results', 'params.txt', 
                                                            'address of output results.')


def _get_oracle_noiselessPBM_propensities(rels, clicks):
  return clicks.sum(0) / rels.sum(0)

def _get_MLE_DCM_lambdas(clicks):
  topk = clicks.shape[1]
  lambdas = np.zeros(topk, dtype=np.float32)
  moving_sum = np.zeros([clicks.shape[0],1])
  
  for col in range(topk-1,-1,-1):
    current_col = clicks[:,col]
    clicked_rows = moving_sum[current_col==1]
    lambdas[col] = 1.0 - (len(clicked_rows[clicked_rows==0]) * 1.0 / len(clicked_rows))
    moving_sum += clicks[:,col][:,None]
    
#   plast = 1 - lambdas
#   plastrel=0.9**(19-np.array(list(range(topk))))
#   lambdas = 1-(plast-plastrel)/(1-plastrel)
#   lambdas[-1] = 1.
  return np.minimum(lambdas, np.ones(len(lambdas)))

def print_one_line(a):
  return str(list(a)).replace(' ', '')

def oracle_propensities(rels, clicks):
  prop2 = _get_oracle_noiselessPBM_propensities(rels, clicks)
  
  with open(FLAGS.output_results, 'a') as f:
    f.write('--train_click_file={} --click_gold_propensities={}\n'.format(
      FLAGS.train_click_file, 
      print_one_line(1/prop2)))

def mle_dcm_lambdas(rels, clicks):
  lam = _get_MLE_DCM_lambdas(clicks)
#   print('DCM:{}'.format(print_one_line_deep(lam)))
#   print('trustPBM:{}'.format(print_one_line(1/prop3)))
  with open(FLAGS.output_results, 'a') as f:
    f.write('--train_click_file={} --click_gold_dcm_lambdas={}\n'.format(
      FLAGS.train_click_file, 
      print_one_line(lam)))

def read_clicks_and_rels(query_file_path, click_file_path, topk):
  with open(query_file_path, 'rb') as f:
    queries = pickle.load(f)
  with open(click_file_path, 'rb') as f:
    clicks = pickle.load(f)
    
  relevances = queries['y'].reshape([-1, topk])
  cl = []
  rel = []
  for i, qid in enumerate(queries['qid']):
    if clicks[qid] is None:
      continue
    reshaped = (clicks[qid].reshape([-1, topk]))
    cl.append(reshaped)
    rel.append(np.ones_like(reshaped) * relevances[i,:])
    
  return np.concatenate(rel, 0), np.concatenate(cl, 0)
  
def main(args):
  rels, clicks = read_clicks_and_rels(query_file_path=os.path.join(FLAGS.train_dir,FLAGS.train_query_file), 
                                      click_file_path=os.path.join(FLAGS.train_dir,FLAGS.train_click_file), 
                                      topk=FLAGS.topk)


  oracle_propensities(rels, clicks)
  mle_dcm_lambdas(rels, clicks)
  
  

if __name__ == '__main__':
  app.run(main)