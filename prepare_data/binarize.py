'''
Created on 18 Nov 2019

@author: aliv
'''


def binarizeRelAndClean(Addr):
  lines = []
  last_qid = -1
  last_lines = []
  rel_sum = 0
  
  query_cnt = 0
  total_query_cnt = 0
  total_lines = 0
  
  for line in open(Addr,'r'):
    total_lines += 1
    splitted = line.split(' ')
    rel = int(splitted[0])
    qid = int(splitted[1][4:])
    
    if qid != last_qid:
      if rel_sum > 0:
        lines += last_lines
        query_cnt += 1
      total_query_cnt += 1
      last_lines = []
      last_real_rels = []
      last_qid = qid
      rel_sum = 0
      
    real_rel = rel
    if rel <= 2:
      rel = 0
    else:
      rel = 1
    
    rel_sum += rel
    
    line = str(rel) + line[1:]
    last_lines.append([real_rel, line])
    last_real_rels.append(real_rel)
  
  # Final query
  if rel_sum > 0:
    lines += last_lines
    query_cnt += 1
  total_query_cnt += 1
  
  print('finished reading {}\n {} docs and {} queries (out of {} and {} original docs and queries)'.format(Addr,len(lines),query_cnt,total_lines,total_query_cnt))
  with open(Addr,'w') as fout:
    for _, line in lines:
      fout.write(line)
  
  with open(Addr+'.real','w') as fout:
    for real_rel, _ in lines:
      fout.write('{}\n'.format(real_rel))
