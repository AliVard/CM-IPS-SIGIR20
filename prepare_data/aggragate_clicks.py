'''
Created on 27 Jan 2020

@author: aliv
'''

import numpy as np
import os

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(  'dir', 'Data/50_2', 
                                                          'directory of click files')

flags.DEFINE_string(  'click_file', 'pbm.set1.valid.b.lgbm_50_2.top20.txt', 
                                                          'address of click logs.')
flags.DEFINE_string(  'aggregated_click_file', 'aggpbm.set1.valid.b.lgbm_50_2.top20.txt', 
                                                          'address of aggregated click logs.')
flags.DEFINE_integer( 'topk', 20, 
                                                          'topk documents shown to user to obtain users\' feedback.')

flags.DEFINE_list(     'min_clicks_list', ['1','3','5'], '')


class AggregateClicks:
  def __init__(self, click_file_path, topk):
    self.aggregated_clicks = {}
    self.topk = topk
    
    click_file = open(click_file_path, 'r')
    
    for line in click_file:
      tokens = line.split(' ')
      current_qid = int(tokens[0][4:])
      clicked_pos = int(tokens[1])
      if current_qid not in self.aggregated_clicks:
        self.aggregated_clicks[current_qid] = np.zeros([self.topk], dtype=np.int32)
      if clicked_pos < self.topk:
        self.aggregated_clicks[current_qid][clicked_pos] += 1
    click_file.close()
      
      
  def save_aggregated(self, aggregated_file_path, min_clicks=1):
    agg_file = open(aggregated_file_path, 'w')
    
    for k, v in self.aggregated_clicks.items():
      for pos in range(self.topk):
        if v[pos] >= min_clicks:
            agg_file.write('qid:{} {}\n'.format(k, pos))
    
    agg_file.close()
    
    
def aggregate_clicks(click_file_path, aggregated_file_path, min_clicks_set, topk):
  AC = AggregateClicks(click_file_path, topk)
  splitted = os.path.splitext(aggregated_file_path)
  for min_click in min_clicks_set:
    AC.save_aggregated('{}.{}{}'.format(splitted[0],min_click,splitted[1]), min_click)
    
    
def main(args):
  aggregate_clicks(os.path.join(FLAGS.dir,FLAGS.click_file), 
                   os.path.join(FLAGS.dir,FLAGS.aggregated_click_file), 
                   list(map(eval,FLAGS.min_clicks_list)), 
                   FLAGS.topk)
  
if __name__ == '__main__':
  app.run(main)