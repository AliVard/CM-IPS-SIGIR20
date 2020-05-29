'''
Created on 18 Nov 2019

@author: aliv
'''
import random
import os


from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string(  'file_path', 'Data/set1.train.b.txt',
                      'light svm file path to be under-sampled.')
flags.DEFINE_bool(    'has_multi_rel', True,
                      'if "file_path" is associated with a ".real" file, containing multi-level relevance labels.')
flags.DEFINE_string(  'output_dir', 'Data/UnderSampled/valid',
                      'output directory. Leave it "None" for a same directory as the "file_path".')
flags.DEFINE_list(    'sample_sizes', ['12800'],
                      'list of sample sizes.')
flags.DEFINE_integer( 'how_many', 5,
                      'how many random files for each sample sizes.')

class UnderSampleClass():
  def __init__(self, file_path, has_multi_rel):
    self._has_multi_rel = has_multi_rel
    
    self.data_list = [] # a list of lists of [real_rel, binarized line] pairs for each qid
    current_id = -1
    
    if self._has_multi_rel:
      for real_rel, line in zip(open(file_path+'.real','r'),open(file_path,'r')):
        id = int(line.split(' ')[1][4:])
        if id != current_id:
          self.data_list.append([])
          current_id = id
        self.data_list[-1].append([real_rel, line])
    else:
      for line in open(file_path,'r'):
        id = int(line.split(' ')[1][4:])
        if id != current_id:
          self.data_list.append([])
          current_id = id
        self.data_list[-1].append([0, line])
      

  
  def underSample(self, sampleSize, output_dir):
    totalSize = len(self.data_list)
    if sampleSize >= totalSize:
      sampleSize = totalSize - 1  # minus one, just for fun
    
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)
    file_number = 0
    
    while os.path.exists(os.path.join(output_dir,'{}_{}.txt'.format(sampleSize, file_number))):
      file_number += 1
      
    outputAddr = os.path.join(output_dir,'{}_{}.txt'.format(sampleSize, file_number))
    
    ids = random.sample(range(totalSize),sampleSize)
    ids.sort()

    with open(os.path.join(output_dir, 'selected_query_ids.txt'), 'a') as ids_file:
      ids_file.write('{}_{}.txt:\t{}\n'.format(sampleSize, file_number, ids))
    
    with open(outputAddr, 'w') as sampledFile:
      for id in ids:
        for _,l in self.data_list[id]:
          sampledFile.write(l)
    
    if self._has_multi_rel:
      with open(outputAddr+'.real', 'w') as sampledFile:
        for id in ids:
          for r,_ in self.data_list[id]:
            sampledFile.write(r)
    
    
    print('{}_{} done!'.format(sampleSize, file_number))

def main(args):
  us = UnderSampleClass(FLAGS.file_path, FLAGS.has_multi_rel)
  output_dir = FLAGS.output_dir
  if output_dir is None:
    output_dir = os.path.dirname(os.path.abspath(FLAGS.file_path))
    
  for sample_size in FLAGS.sample_sizes:
    for _ in range(FLAGS.how_many):
      us.underSample(int(sample_size), output_dir)


if __name__ == '__main__':
  app.run(main)