'''
Created on 20 Nov 2019

@author: aliv
'''
import os
import sys
import numpy as np
import random
from sklearn.datasets import load_svmlight_file
import pickle

# import matplotlib.pyplot as plt

from absl import app
from absl import flags

SYS_PATH_APPEND_DEPTH = 3
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)
from CM_IPS.prepare_data import common_functions


if __name__ == '__main__':
  FLAGS = flags.FLAGS
  
  flags.DEFINE_string(  'data_dir', 'Data/set1.train.b.lgbm_50_2.top20.pkl',
                        'data directory')
  flags.DEFINE_string(  'output_dir', None,
                        'output directory. "None" for the same as "data_dir"')
  flags.DEFINE_string(  'clicks_count', '2**18',
                        'clicks count')
  flags.DEFINE_integer( 'topk', 20,
                        'topk')
  flags.DEFINE_string(  'model', 'dcm_1.0',
                        'generate clicks based on model: ideal, pbm, dcm, dbn, trustpbm')
  flags.DEFINE_string(  'positive_noise', '0.0', 'probability of not attraction given relevancy')
  flags.DEFINE_string(  'negative_noise', '0.0', 'probability of attraction given non-relevancy')

def write_clicks(sessions, output_path, click_fn, clicks_count):
  cnt = 0
  y = sessions['y']
  qid = sessions['qid']
  y = y.reshape([qid.shape[0], FLAGS.topk])
  clicks = {}
  for id in qid:
    clicks[id] = []
  
  while cnt < clicks_count:
    id = np.random.choice(qid.shape[0])
    clicked = click_fn(y[id,:])
    cnt += sum(clicked)
    clicks[qid[id]].append(clicked)
  
  for id in qid:
    if len(clicks[id]) > 0:
      clicks[id] = np.concatenate(clicks[id],0)
    else:
      clicks[id] = None
      
  with open(output_path, 'wb') as f:
    pickle.dump(clicks, f, 2)
  return cnt
  

class PBM_binary:
  def __init__(self, observe_probs, pos_noise, neg_noise):
    self._observe_probs = np.array(observe_probs)
    self._pos_noise = pos_noise
    self._neg_noise = neg_noise
    
  def clicked(self, y):
    prob = self._observe_probs
    mask = np.ones_like(prob)
    mask[y==0.] = self._neg_noise
    mask[y==1.] = 1 - self._pos_noise
    
      
    return np.random.binomial(1, mask * prob)

  @staticmethod
  def get_params(model_name):
    splitted = model_name.split('_')
    eta = 1.0
    if len(splitted) > 1:
      eta = float(splitted[1])
      
    return eta
  
  
  @staticmethod
  def get_generator(model_name):
    eta = PBM_binary.get_params(model_name)
    def __generator(sessions, output_path, clicks_count):
      return PBM_binary.generate_clicks_binary(sessions, output_path, clicks_count, eta)
    
    return __generator
  
  
  @staticmethod
  def generate_clicks_binary(sessions, output_path, clicks_count, eta):
    click_model = PBM_binary([((1.0/(i+1)) ** eta) for i in range(FLAGS.topk)], eval(FLAGS.positive_noise), eval(FLAGS.negative_noise))
    
    write_clicks(sessions, output_path, click_model.clicked, clicks_count)
  
class DCM_binary:
  def __init__(self, gamma, pos_noise, neg_noise):
    self._gamma = np.array(gamma)
    self._pos_noise = pos_noise
    self._neg_noise = neg_noise
   
  def clicked(self, y):
    clicks = np.zeros_like(y)
    abandoned = False
    for i in range(y.shape[0]):
      if abandoned:
        break
      if y[i] == 0:
        clicks[i] = random.random() < self._neg_noise
      else:
        clicks[i] = random.random() < 1 - self._pos_noise
      if clicks[i]:
        abandoned = random.random() < 1 - self._gamma[i]

    return clicks
  
  @staticmethod
  def generate_clicks_binary(sessions, output_path, clicks_count, mult, eta):
    click_model = DCM_binary([(mult * ((1.0/(i+1)) ** eta)) for i in range(FLAGS.topk)], eval(FLAGS.positive_noise), eval(FLAGS.negative_noise))
    
    write_clicks(sessions, output_path, click_model.clicked, clicks_count)
    

  @staticmethod
  def get_params(model_name):
    splitted = model_name.split('_')
    mult = 1.0
    eta = 1.0
    if len(splitted) >= 2:
      mult = float(splitted[1])
    if len(splitted) == 3:
      eta = float(splitted[2])
      
    return mult, eta
  
  @staticmethod
  def get_generator(model_name):
    mult, eta = DCM_binary.get_params(model_name)
    def __generator(sessions, output_path, clicks_count):
      return DCM_binary.generate_clicks_binary(sessions, output_path, clicks_count, mult, eta)
    
    return __generator

def read_sessions_binary(input_path):
  with open(input_path, 'rb') as f:
    sessions = pickle.load(f)
  return sessions


def wrapper(args):
  for model in ['dcm_{}_{}'.format(i,j) for i in ['0.6','0.8'] for j in ['0.5','1.0','2.0']]:
    FLAGS.model = model
    main(args)
    
def main(args):
  clicks_count = int(eval(FLAGS.clicks_count))
  
  # default generator: ideal
  generator = None
  model_name = FLAGS.model
  
#   print('+:{}, -:{}'.format(eval(FLAGS.positive_noise), eval(FLAGS.negative_noise)))
  model_class = None
  if model_name.startswith('pbm'):
    model_class = PBM_binary
  elif model_name.startswith('dcm'):
    model_class = DCM_binary

  if model_class is not None:
    generator = model_class.get_generator(model_name)
  
  data_dir, data_files = common_functions.get_files(FLAGS.data_dir)
  
  if FLAGS.output_dir:
    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
  else:
    output_dir = data_dir
    
  if generator is not None:
    for file in data_files:
      sessions = read_sessions_binary(os.path.join(data_dir, file))
  #     print('{} reading finished!'.format(file))
  
      generator( sessions, 
                 os.path.join(output_dir, FLAGS.clicks_count +'.' + model_name + '.' + file),
                 clicks_count)
  #     print('file {} with {} clicks done!'.format(file, clicks))

if __name__ == '__main__':
  app.run(main)
#   app.run(wrapper)

#   pos_trust = [0.99-(i/100.0) for i in range(20)]
#   neg_trust = [0.75/(i+1) for i in range(20)]
#   x = list(range(1,21))
#   y = [p/(i+1) for (i,p) in enumerate(pos_trust)]
#   plt.plot(x,y,label='r=1')
#   plt.legend()
#   y = [p/(i+1) for (i,p) in enumerate(neg_trust)]
#   plt.plot(x,y,label='r=0')
#   plt.legend()
#   plt.savefig('trustpbm_probs.png')
      
