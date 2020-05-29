'''
Created on 6 Nov 2019

@author: aliv
'''

from __future__ import print_function
# from future.utils import raise_

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CM_IPS.Losses import m_loss_from_str

import logging


def sigmoid_prob(logits):
  return 1.*tf.sigmoid(1.*(logits - tf.reduce_mean(logits, -1, keepdims=True)))

def min_max_prob(logits):
  e = tf.exp(logits)
  e = e - tf.reduce_min(e,-1,keepdims=True)
  return 1. * e / tf.reduce_max(e, -1, keepdims=True)

class RelNet:
  def __init__(self, 
               layers_size, embed_size, batch_size,
               forward_only, drop_out_probs,
               learning_rate, 
               loss_function_str, rank_list_size, 
               max_gradient_norm, l2_loss,
               batch_normalize,
               gold_propensities, 
               gold_dcm_lambdas, propensity_clip,
               perplexity_prob_fn,
               logger=None):
    '''
    args:
      rank_list_size: k as in top-k docs
      optimizer: either 'adagrad' or 'grad'
    '''

    if isinstance(drop_out_probs, list):
      self._drop_out_probs = np.zeros(len(layers_size), dtype=np.float32)
      for i in range(min(len(layers_size),len(drop_out_probs))):
        self._drop_out_probs[i] = drop_out_probs[i]
    else:
      self._drop_out_probs = np.ones(len(layers_size))*drop_out_probs

    self._hidden_layer_sizes = list(map(int, layers_size))
    
    self.batch_size = batch_size
    self._embed_size = embed_size
    self._rank_list_size = rank_list_size
    
    self.graph = tf.Graph()
    
    self.gold_propensities = np.array(list(map(float, eval(gold_propensities)))) if gold_propensities is not None else None
    self.gold_dcm_lambdas = np.array(list(map(float, eval(gold_dcm_lambdas)))) if gold_dcm_lambdas is not None else None
    self.propensity_clip = propensity_clip
    
    self.correct_clicks = self.no_correction
    if self.gold_propensities is not None:
      self.gold_propensities = np.minimum(self.gold_propensities, propensity_clip*np.ones_like(self.gold_propensities))
      self.correct_clicks = self.pbm_correction
    elif self.gold_dcm_lambdas is not None:
      self.gold_dcm_lambdas[self.gold_dcm_lambdas<0.001]=0.001
      self.gold_dcm_lambdas[-1] = 1
      self.correct_clicks = self.dcm_correction
    
    with self.graph.as_default():
      
      self.global_step = tf.Variable(0, trainable=False)
      
  
      self.tf_input_embeddings = tf.placeholder(tf.float32, shape=(None, self._embed_size), name='rel/embedding')
      self.tf_relevance_labels = tf.placeholder(tf.float32, shape=(None, self._rank_list_size), name='rel/labels')
      self.ltf_dropout_rate    = []
      for dropout_rate in self._drop_out_probs:
        self.ltf_dropout_rate.append(tf.placeholder_with_default(dropout_rate, shape=()))

      
      # Build model
      self.output = self.network(forward_only, batch_normalize)
      
      params = tf.trainable_variables()
        
      self.loss = self.generate_loss(output=self.output,  labels=self.tf_relevance_labels,
                                     loss_function_str=loss_function_str)
      tf.summary.scalar('loss',self.loss)
      
      
#       self.perplexity = self.generate_perplexity(self.output, self.tf_relevance_labels, self.propensities, perplexity_prob_fn)
      
      if not forward_only:
        opt = tf.train.AdagradOptimizer(learning_rate)
        self.gradients = tf.gradients(self.loss, params)
        if max_gradient_norm > 0:
          self.clipped_gradients, self.norm = tf.clip_by_global_norm(self.gradients,
                                     max_gradient_norm)
          self.updates = opt.apply_gradients(zip(self.clipped_gradients, params),
                         global_step=self.global_step)
        else:
          self.norm = None #tf.norm(self.gradients)
          self.updates = opt.apply_gradients(zip(self.gradients, params),
                         global_step=self.global_step)
      
      self.summary = tf.summary.merge_all()
      self.saver = tf.train.Saver(tf.global_variables())
    
    
  def no_correction(self, clicks):
    y = []
    for q_clicks in clicks:
      if q_clicks is None:
        y.append(np.zeros(self._rank_list_size, dtype=np.float32))
      else:
        reshaped = q_clicks.reshape([-1, self._rank_list_size])
        y.append(np.mean(reshaped,0))
    return np.concatenate(y,0).reshape([-1, self._rank_list_size])
  
  def pbm_correction(self, clicks):
    y = self.no_correction(clicks)
    return  y * self.gold_propensities

  def get_propensities_from_dcm_lambdas(self, y):
    e = np.log(1-(y*(1-self.gold_dcm_lambdas)))
    sum_cols = np.zeros([len(self.gold_dcm_lambdas), len(self.gold_dcm_lambdas)], dtype=np.float32)
    for row in range(len(self.gold_dcm_lambdas)-1):
      sum_cols[row,row+1:] = 1.0
    prop = np.exp(-np.matmul(e,sum_cols))
    
    return np.minimum(prop, self.propensity_clip*np.ones_like(prop))
  
  def dcm_correction(self, clicks):
    y = []
    for q_clicks in clicks:
      reshaped = q_clicks.reshape([-1,self._rank_list_size])
      propensity = self.get_propensities_from_dcm_lambdas(reshaped)
      y.append(np.mean(reshaped * propensity, 0))
    return np.concatenate(y,0).reshape([-1, self._rank_list_size])
      
  def network(self, forward_only, batch_normalize):
    
    tf_output = self.tf_input_embeddings
    current_size = self._embed_size
    
    for layer in range(len(self._hidden_layer_sizes)):
      fan_in = current_size
      fan_out = self._hidden_layer_sizes[layer]
#       glorot_uniform_initializer as is default in tf.get_variable()
      r = np.sqrt(6.0/(fan_in+fan_out))
#       tf_w = tf.Variable(tf.random_normal([current_size, self._hidden_layer_sizes[layer]], stddev=0.1), name='rel/w_{}'.format(layer))
      tf_w = tf.Variable(tf.random_uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float32), name='rel/w_{}'.format(layer))

      
      tf_b = tf.Variable(tf.constant(0.1, shape=[fan_out]), name='rel/b_{}'.format(layer))
      
      # x.w+b
      tf_output_tmp = tf.nn.bias_add(tf.matmul(tf_output, tf_w, name='rel/mul_{}'.format(layer)), tf_b, name='rel/affine_{}'.format(layer))
      
      # normalize
      if batch_normalize:
        tf_output_tmp = tf.layers.batch_normalization(tf_output_tmp, training=forward_only, name='rel/batch_norm_{}'.format(layer))
        
      # activation: elu
      tf_output_tmp = tf.nn.elu(tf_output_tmp, name='rel/elu_{}'.format(layer))
      
      # residual link
      if fan_in == fan_out:
        tf_output = tf.add(tf_output, tf_output_tmp)
      else:
        tf_output = tf_output_tmp
      
      # generalization: drop_out
      if self._drop_out_probs[layer] > 0.0 and not forward_only:
#         With probability rate elements are set to 0. 
#         The remaining elemenst are scaled up by 1.0 / (1 - rate), so that the expected value is preserved.
        tf_output = tf.nn.dropout(tf_output, rate=self.ltf_dropout_rate[layer], name = 'rel/drop_out_{}'.format(layer))

      current_size = self._hidden_layer_sizes[layer]
    
    
    # Output layer
  
    fan_in = self._hidden_layer_sizes[-1]
    fan_out = 1
    r = np.sqrt(6.0/(fan_in+fan_out))
#     tf_w = tf.Variable(tf.random_normal([self._hidden_layer_sizes[-1], 1], stddev=0.1), name='rel/w_last')
    tf_w = tf.Variable(tf.random_uniform([fan_in, fan_out], minval=-r, maxval=r, dtype=tf.float32), name='rel/w_last')
    tf_b = tf.Variable(tf.constant(0.1, shape=[fan_out]), name='rel/b_{}'.format('last'))
    
    tf_output = tf.nn.bias_add(tf.matmul(tf_output, tf_w), tf_b, name='rel/affine_last')
    
    tf_output = tf.reshape(tf_output, [-1, self._rank_list_size])
    return tf_output

  def relevance_prob(self, output):
    return tf.nn.sigmoid(output - tf.reduce_mean(output, -1, keepdims=True))
  
  def get_propensities(self, output, click_probs):
#     prop = 1.0/(click_probs+1e-4)
    prop = (self.relevance_prob(output)+1.0+1e-4)/(click_probs+1e-4)/2
    
    return tf.maximum(tf.ones_like(prop), prop)
  
  def generate_loss(self, output, labels, loss_function_str):    
    loss_fn = m_loss_from_str(loss_function_str)
    
    loss = loss_fn(y_true=labels, y_pred=output)

    return loss


  def generate_perplexity(self, output, labels, propensities, perplexity_prob_fn=None):
    logits_to_prob = tf.nn.softmax
    if perplexity_prob_fn == 'sigmoid':
      logits_to_prob = sigmoid_prob
    elif perplexity_prob_fn == 'min_max':
      logits_to_prob = min_max_prob
#     probs = ((tf.nn.softmax(self.output) + sigmoid_prob(self.output)) / 2.) / propensities
    probs = logits_to_prob(self.output) / propensities
    probs = tf.minimum(probs, 0.99 * tf.ones_like(probs))
    probs = tf.maximum(probs, 0.01 * tf.ones_like(probs))
    perp = labels * tf.log(probs) / np.log(2.)
    perp += (1. - labels) * tf.log(1. - probs) / np.log(2.)
    perp = tf.reduce_mean(perp,0)
    return 2**(-perp)



  def step(self, session, input_feed, forward_only):
    if not forward_only:
      output_feed = [self.loss, self.updates]  
    else:
      for tf_dropout in self.ltf_dropout_rate:
        input_feed[tf_dropout.name] = 0.
      output_feed = [self.loss, self.output]
    
    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[0]
    else:
      return outputs[0], outputs[1]  # loss, outputs
      
  def predict(self, session, input_feed):
    for tf_dropout in self.ltf_dropout_rate:
      input_feed[tf_dropout.name] = 0.
    outputs = session.run([self.output], input_feed)
    return outputs[0]
  
  def get_perplexity(self, session, input_feed):
  
    perp = session.run([self.perplexity], input_feed)
    return perp[0]
    
  # get_epoch is only used for test.
  def get_epoch(self, mDataset):
    input_feeds = {}

    input_feeds[self.tf_input_embeddings.name] = mDataset.X
    input_feeds[self.tf_relevance_labels.name] = mDataset.y.reshape([-1,self._rank_list_size])
    
    return input_feeds
  
  def get_next_batch(self, mDataset):
    indexes = mDataset.get_random_indexes(self.batch_size)
    mDataset.load_batch(indexes)
    input_feeds = {}
    
    input_feeds[self.tf_input_embeddings.name] = mDataset.x_train   # shape=(None, self._embed_size)
    input_feeds[self.tf_relevance_labels.name] = self.correct_clicks(mDataset.y_train)   # shape=(None, self._rank_list_size)
   
    return input_feeds

  def get_seq_batch(self, seq_index, mDataset):   
    if seq_index >= mDataset.samples_size:
      raise Exception('sequential batch start index exceeds total number of samples!')
  
    start = seq_index
    seq_index += self.batch_size
    end = seq_index if seq_index <= mDataset.samples_size else mDataset.samples_size
    
    mDataset.load_batch(list(range(start,end)))
    input_feeds = {}
  
    input_feeds[self.tf_input_embeddings.name] = mDataset.x_train   # shape=(None, self._embed_size)
    input_feeds[self.tf_relevance_labels.name] = mDataset.y_train   # shape=(None, self._rank_list_size)
    
    return input_feeds
  
















  