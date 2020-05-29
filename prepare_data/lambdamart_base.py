'''
Created on 18 Nov 2019

@author: aliv
'''

import lightgbm as lgb
import numpy as np
from sklearn.datasets import load_svmlight_file
import os
import sys


from absl import app
from absl import flags


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CM_IPS.prepare_data import metrics, common_functions

FLAGS = flags.FLAGS

flags.DEFINE_string(  'train_data', 'Data/UnderSampled/train/11473*.txt', 
                      'train data path.')
flags.DEFINE_string(  'valid_data', 'Data/UnderSampled/valid/1713_0.txt', 
                      'validation data path.')
flags.DEFINE_string(  'test_data', 'Data/set1.train.b.txt',
                      'path of data to be "predict"ed.')

flags.DEFINE_string(  'model_dir', 'lambdaMART/lgbm_11*', 
                      'with --nopredict (default): LambdaMART model directory for saving trained model.'
                      'with --predict: LambdaMART model directory for loading all trained models named "lgbm_*.txt" inside.')

flags.DEFINE_integer( 'early_stopping_rounds', 1000,
                      'early_stopping_rounds for lightgbm\'s LGBMRanker().fit() function.')

flags.DEFINE_bool(    'predict', True,
                      'train ("predict"=False) or test("predict"=True)')
flags.DEFINE_list(    'eval_at', ['1', '3', '5', '10'],
                      'list of "k" values for evaluating NDCG@k.')

def lambdarank(trPath, tstPath, model_path, early_stopping_rounds):
  
  X_train, y_train, q_train = load_svmlight_file(trPath, query_id=True)
  X_test, y_test, q_test = load_svmlight_file(tstPath, query_id=True)
  _, q_train = np.unique(q_train,return_counts=True)
  _, q_test = np.unique(q_test,return_counts=True)

  gbm = lgb.LGBMRanker()
#   gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],
#           eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=100, verbose=False,
#           callbacks=[lgb.reset_parameter(learning_rate=lambda x: max(0.01, 0.1 - 0.01 * x))])
  
  
  gbm.fit(X_train, y_train, group=q_train, eval_set=[(X_test, y_test)],
          eval_group=[q_test], eval_at=[1, 3], early_stopping_rounds=early_stopping_rounds, verbose=False)
  
  gbm.booster_.save_model(model_path)
  
  return gbm
  
def predict_one_model(model_path, X_test, y_test, q_test, eval_at):
  booster = lgb.Booster(model_file=model_path)
  y_pred = booster.predict(X_test)
  metric = metrics.LTRMetrics(y_test,q_test,y_pred)
#   print(metric.queryCount())
  return [metric.NDCG(k) for k in eval_at]


def train(trPath, valPath, modelDir, early_stopping_rounds):
  if not os.path.exists(modelDir):
    os.makedirs(modelDir)
  
  
  data_dir, data_files = common_functions.get_files(trPath)
  
  if not os.path.isfile(trPath) and trPath == valPath:
    raise ValueError('when using multiple models training by specifying a directory instead of a file, train dir cannot be the same as validation dir.')

  for file in data_files:
    if not os.path.isfile(valPath):
      val_path = os.path.join(valPath, file)
    else:
      val_path = valPath
    modelFile = 'lgbm_' + file
    if not os.path.exists(val_path):
      continue
    try:
      lambdarank(os.path.join(data_dir, file), val_path, os.path.join(modelDir, modelFile), early_stopping_rounds)
      print('finished training {} model!'.format(file))
    except:
      pass

  
def predict(model_dir, test_path, eval_at):
  X_test, y_test, q_test = load_svmlight_file(test_path, query_id=True)
  _, q_test = np.unique(q_test,return_counts=True)
  
  print('test file read!')
  
  
  data_dir, data_files = common_functions.get_files(model_dir)
  
  for file in data_files:
    if os.path.isfile(os.path.join(data_dir, file)):
      if str.startswith(file,'lgbm_'):
        try:
          result = predict_one_model(os.path.join(data_dir, file), X_test, y_test, q_test, eval_at)
          print('{} -> {}'.format(file, result))
        except:
          pass


def main(args):
  if FLAGS.predict:
    predict(FLAGS.model_dir, FLAGS.test_data, list(map(int,FLAGS.eval_at)))
  else:
    train(FLAGS.train_data, FLAGS.valid_data, FLAGS.model_dir, FLAGS.early_stopping_rounds)
    
if __name__ == '__main__':
  app.run(main)
  