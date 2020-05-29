'''
Created on 22 Nov 2019

@author: aliv
'''
from __future__ import print_function
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS

import sys
import time
import os

SYS_PATH_APPEND_DEPTH = 2
SYS_PATH_APPEND = os.path.abspath(__file__)
for _ in range(SYS_PATH_APPEND_DEPTH):
  SYS_PATH_APPEND = os.path.dirname(SYS_PATH_APPEND)
sys.path.append(SYS_PATH_APPEND)

from CM_IPS.data_utils_agg import read_relevance_data
from CM_IPS.RelevanceProbs import RelNet
from CM_IPS.prepare_data.metrics import eval_predictions, eval_output

import logging



NON_DEFINED_FLAGS = set()
for k, _ in FLAGS.__flags.items():
  NON_DEFINED_FLAGS.add(k)

DEFINED_FLAGS_CLICKS = set()
for k, _ in FLAGS.__flags.items():
  if not k in NON_DEFINED_FLAGS:
    DEFINED_FLAGS_CLICKS.add(k)
    NON_DEFINED_FLAGS.add(k)

flags.DEFINE_string(  'click_gold_propensities', None,
                                                          'this can be set for ground truth propensities.')
flags.DEFINE_string(  'click_gold_dcm_lambdas', "[1./(i+1) for i in range(1,21)]",
                                                          'in case of "--click_gold_propensities=None" this can be set for ground truth dcm lambdas.')
flags.DEFINE_float(   'click_propensity_clip', 100,
                                                          'propensity clip.')

    
# RELEVANCE MODEL specific --------------------------------------------------------------------------------------------------------------
flags.DEFINE_list(    'rel_hidden_layer_size', [512, 256, 128], 
                                                          'list of hidden layer sizes')
flags.DEFINE_integer( 'rel_batch_size', 128, 
                                                          'batch size')
flags.DEFINE_list(    'rel_drop_out_probs', [0.0, 0.1, 0.1], 
                                                          'layer specific drop out probabilities. It has to have similar length with "hidden_layer_size" flag.')
flags.DEFINE_string(  'rel_learning_rate', '4e-3', 
                                                          'learning rate')
flags.DEFINE_float(   'rel_max_gradient_norm', 50.0, 
                                                          'if > 0, this value is used to clip the gradients.')
flags.DEFINE_float(   'rel_l2_loss', 0.0, 
                                                          'used for regularization with l2 norm. This is the coefficient!')
flags.DEFINE_string(  'rel_loss_fn', 'softmax',
                                                          '[w*]func[(param)+w*func2(param)+...] where func can be "softmax", "sigmoid" or "pair_sigmoid"')
flags.DEFINE_boolean( 'rel_batch_normalize', False, 
                                                          'apply batch normalization at each layer')
flags.DEFINE_boolean( 'rel_fresh', True, 
                                                          'set for ignoring the chekpoint model in "ckpt_dir" directory')

flags.DEFINE_string(  'rel_ckpt_dir', '', 
                                                          'directory of check points train files (if any)')


DEFINED_FLAGS_RELS = set()
for k, _ in FLAGS.__flags.items():
  if not k in NON_DEFINED_FLAGS:
    DEFINED_FLAGS_RELS.add(k)
    NON_DEFINED_FLAGS.add(k)
    
# SHARED FLAGS --------------------------------------------------------------------------------------------------------------
# file addresses:
flags.DEFINE_string(  'train_dir', 'Data/', 
                                                          'directory of train files')
flags.DEFINE_string(  'train_query_file', 'set1.train.b.lgbm_50_2.top20.pkl', 
                                                          'address of top-k document embeddings retrieved for train queries.')
flags.DEFINE_string(  'train_click_file', 'dcm_1.0.set1.train.b.lgbm_50_2.top20.pkl', 
                                                          'address of train click logs.')
flags.DEFINE_string(  'test_dir', 'Data/', 
                                                          'directory of test files')
flags.DEFINE_string(  'test_query_file', 'set1.test.b.lgbm_50_2.top20.pkl', 
                                                          'address of top-k document embeddings retrieved for test queries.')
flags.DEFINE_string(  'test_click_file', 'pbm_1.0.test.b.lgbm_50_2.top20.pkl', 
                                                          'address of click logs for test queries.')
flags.DEFINE_string(  'file_log_path', 'file.log', 
                                                          'path for logging.')
# data params:
flags.DEFINE_string(  'click_count', '2**16', 
                                                          'number of clicks to train on. accepts "eval"able expressions like "1e6" or "2**20" as well.')
flags.DEFINE_integer( 'topk', 20, 
                                                          'topk documents shown to user to obtain users\' feedback.')

# other params:
flags.DEFINE_integer( 'max_train_iteration', 200, 
                                                          'Limit on the iterations of training.')
flags.DEFINE_integer( 'steps_per_checkpoint', 50, 
                                                          'How many training steps to do per checkpoint.')
flags.DEFINE_boolean( 'predict', False,  
                                                          'set for predicting test data using learned model in "ckpt_dir" directory')

flags.DEFINE_boolean( 'train_and_predict', True,  
                                                          'set for training on train data and then predicting test data using learned model in "ckpt_dir" directory')
flags.DEFINE_string(  'train_and_predict_output', 'train_and_predict_output.txt', 
                                                          'comma separated output file for hyper parameter tuning purposes.')
flags.DEFINE_string(  'perplexity_output', 'perplexity_output.txt', 
                                                          'comma separated output file for hyper parameter tuning purposes.')
flags.DEFINE_string(  'perplexity_prob_fn', 'softmax', 
                                                          'function used for converting logits to probs for perplexity. either "softmax", "min_max" or "sigmoid".')

flags.DEFINE_string(  'slurm_job_id', '0', 
                                                          'job id (and task id in case of array) from slurm')




DEFINED_FLAGS = set()
for k, _ in FLAGS.__flags.items():
  if not k in NON_DEFINED_FLAGS:
    DEFINED_FLAGS.add(k)

click_global_step = 0
rel_global_step = 0

def create_rel_model(data_set, forward_only, logger=None):
  model = RelNet(    layers_size=FLAGS.rel_hidden_layer_size,
                     embed_size=data_set.embed_size, batch_size=FLAGS.rel_batch_size,
                     forward_only=forward_only, drop_out_probs=FLAGS.rel_drop_out_probs,
                     learning_rate=eval(FLAGS.rel_learning_rate), 
                     loss_function_str=FLAGS.rel_loss_fn, rank_list_size = FLAGS.topk, 
                     max_gradient_norm=FLAGS.rel_max_gradient_norm, l2_loss=FLAGS.rel_l2_loss,
                     batch_normalize=FLAGS.rel_batch_normalize,
                     gold_propensities=FLAGS.click_gold_propensities, 
                     gold_dcm_lambdas=FLAGS.click_gold_dcm_lambdas, propensity_clip=FLAGS.click_propensity_clip,
                     perplexity_prob_fn=FLAGS.perplexity_prob_fn,
                     logger=logger)
  return model

def initialize_model(session, checkpoint_path, fresh, model, logger):
  ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
#   ckpt = None
  model.session = session
  if (FLAGS.predict or not fresh) and ckpt:
    logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    logger.info("Created model with fresh parameters.")
    with model.graph.as_default():
      init_globals = tf.global_variables_initializer()
    session.run(init_globals)

def train_model(  model_name, checkpoint_path, fresh, 
                  train_set,
                  logger=None):

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  
  model = create_rel_model(train_set, False, logger)
  
  sess = tf.Session(config=config, graph=model.graph)
  # Create model.
  logger.info("Creating %s model ..." % model_name)
  initialize_model(sess, checkpoint_path, fresh, model, logger=logger)

  # This is the training loop.
  step_time, loss = 0.0, 0.0
  current_step = model.global_step.eval(session=sess)
#     best_loss = None
  while True:
    start_time = time.time()
    
    input_feed = model.get_next_batch(train_set)

    step_loss = model.step(sess, input_feed, False)

    step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
    loss += step_loss / FLAGS.steps_per_checkpoint
    current_step += 1
    
    if current_step < 10:
      logger.info("%s model: global step %d loss %.9f" % 
                  (model_name, model.global_step.eval(session=sess), step_loss))
      
    if current_step % FLAGS.steps_per_checkpoint == 0:
      logger.info("%s model: global step %d step-time %.2f loss %.9f" % 
                  (model_name, model.global_step.eval(session=sess), step_time, loss))
            
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)
      
      
      if loss == float('inf'):
        break

      step_time, loss = 0.0, 0.0
      sys.stdout.flush()

    if current_step > FLAGS.max_train_iteration:
      break
        
    
  model.saver.save(sess, checkpoint_path, global_step=model.global_step)
    
  return model


def train(logger, close_session=True):
  
  click_count = int(eval(FLAGS.click_count))

  # Prepare data.
  logger.info("Reading data from %s for relevance model." % FLAGS.train_dir)
  
  rel_train_set = read_relevance_data(os.path.join(FLAGS.train_dir, FLAGS.train_query_file), 
                                                 os.path.join(FLAGS.train_dir, FLAGS.train_click_file), 
                                                 click_count, FLAGS.topk)
  
  rel_model = train_model(  model_name='relevance', 
                            checkpoint_path=os.path.join(FLAGS.rel_ckpt_dir, "relNet.ckpt"), 
                            fresh=FLAGS.rel_fresh, 
                            train_set=rel_train_set, 
                            logger=logger) 

    
  rel_model.rel_global_step = rel_model.global_step.eval(session=rel_model.session)
  
  
  if close_session:
    rel_model.session.close()
  
  return rel_model

def test(rel_model=None, logger=None, close_session=True):
  #   tf.debugging.set_log_device_placement(True)
  
  # Prepare data.
  logger.info("Reading data in %s" % FLAGS.test_dir)
  
  test_set = read_relevance_data(os.path.join(FLAGS.test_dir, FLAGS.test_query_file), 
                                   None, 
                                   0, FLAGS.topk)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
 
  
  if rel_model is None:
    model = create_rel_model(test_set, True, None)
  
    sess = tf.Session(config=config, graph=model.graph)
    # Create model.
    logger.info("Creating model...")
    initialize_model(sess, os.path.join(FLAGS.rel_ckpt_dir, "relNet.ckpt"), False, model, logger=logger)
  else:
    model = rel_model
    sess = rel_model.session


  # This is the training loop.
  step_time = 0.0
  
  start_time = time.time()
    
  input_feed = model.get_epoch(test_set)

  predicted = model.predict(sess, input_feed)

  step_time += (time.time() - start_time)
  
  output_path = os.path.join(FLAGS.rel_ckpt_dir,'rel_predictions_' + FLAGS.test_query_file)
  
  y_pred = predicted.reshape([-1, 1])
#     print(y_pred.shape)
  y_true = test_set.y.reshape([-1, 1])
#     print(y_true.shape)
  joint_ys = np.concatenate((y_pred, y_true), axis=1)
#     print(joint_ys.shape)
  np.savetxt(output_path, joint_ys, fmt='%.5f, %.1f')
  print('output saved in %s' % output_path)
  
  results = eval_output(y_true=y_true[:,0], y_pred=y_pred[:,0], query_counts=FLAGS.topk)
  
  print(results)
  
  if close_session:
    model.session.close()
  
  return results

def perplexity(rel_model=None, logger=None):
  
  click_count = int(eval(FLAGS.click_count))
  click_test_set = read_relevance_data( os.path.join(FLAGS.test_dir, FLAGS.test_query_file),
                                        os.path.join(FLAGS.test_dir, FLAGS.test_click_file),
                                        int(click_count * 0.35), FLAGS.topk)
  
  
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
 
  
  if rel_model is None:
    model = create_rel_model(click_test_set, True, None)
  
    sess = tf.Session(config=config, graph=model.graph)
    # Create model.
    logger.info("Creating model...")
    initialize_model(sess, os.path.join(FLAGS.rel_ckpt_dir, "relNet.ckpt"), False, model, logger=logger)
  else:
    model = rel_model
    sess = rel_model.session

  it = 0
  perplexities = []

  while it < click_test_set.samples_size - model.batch_size:
    input_feed = model.get_seq_batch(it, click_test_set)
    perp = model.get_perplexity(sess, input_feed)
    it += model.batch_size
    perplexities.append(perp)
  if click_test_set.samples_size - it > 0:
    input_feed = model.get_seq_batch(it, click_test_set)
    perp = model.get_perplexity(sess, input_feed)
    perplexities.append(perp * model.batch_size / (click_test_set.samples_size - it))
    
  perplexities = list(np.mean(np.array(perplexities),axis=0))
  
  print(perplexities)
  
  model.session.close()
    
  return perplexities
      
def my_serialize(v):
  if v.value is not None:
    return v.serialize()
  else:
    return '--{}=None'.format(v.name)

def train_and_test(logger):
  rel_model = train(logger=logger, close_session=False)
  results = test(rel_model, logger=logger, close_session=False)
#   perp = perplexity(rel_model, logger=logger)
  separator_char = ''
  general_info = ''
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS_CLICKS:
      general_info += '{}"{}"'.format(separator_char, my_serialize(v)[2:])
      separator_char = ', '
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS_RELS:
      general_info += '{}"{}"'.format(separator_char, my_serialize(v)[2:])
      separator_char = ', '
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS:
      general_info += '{}"{}"'.format(separator_char, my_serialize(v)[2:])
      separator_char = ', '
    
  with open(FLAGS.train_and_predict_output, 'a') as fo:
    fo.write('{}, {}, "rel_global_step={}", "nDCG={}"\n'.format(
                                                                                    FLAGS.slurm_job_id, 
                                                                                    general_info, 
                                                                                    rel_model.rel_global_step, 
                                                                                    results))
#   with open(FLAGS.perplexity_output, 'a') as fo:
#     fo.write('{}, {}, "rel_global_step={}", "perplexity={}"\n'.format(
#                                                                                     FLAGS.slurm_job_id, 
#                                                                                     general_info,
#                                                                                     rel_model.rel_global_step, 
#                                                                                     perp))
  
def main(args):
  
  logger = logging.getLogger('AliV')
  
  logger.info('Last modified: Thu Feb 20 01:28:59 2020')
  
  f_handler = logging.FileHandler(FLAGS.file_log_path)
  f_handler.setLevel(logging.DEBUG)
  f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  f_handler.setFormatter(f_format)
  
  logger.addHandler(f_handler)

  general_info = '\ntensorflow {}\n\n'.format(tf.__version__)
  
   
  general_info += '  click related flags:\n'
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS_CLICKS:
      general_info += '    {}\n'.format(my_serialize(v)) #''    - {} : {}\n'.format(k,v.value)

  general_info += '\n\n'
  
  general_info += '  relevance related flags:\n'
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS_RELS:
      general_info += '    {}\n'.format(my_serialize(v)) #''    - {} : {}\n'.format(k,v.value)

  general_info += '\n\n'
  
  general_info += '  shared flags:\n'
  for k, v in FLAGS.__flags.items():
    if k in DEFINED_FLAGS:
      general_info += '    {}\n'.format(my_serialize(v)) #''    - {} : {}\n'.format(k,v.value)

  general_info += '\n\n'
  
  logger.info(general_info)
    

  if FLAGS.train_and_predict:
    train_and_test(logger=logger)
  elif FLAGS.predict:
    test(None, logger=logger)
  else:
    train(logger=logger)
  
if __name__ == '__main__':
  app.run(main)