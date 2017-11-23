#coding:utf-8

from config import path,model_result

def run(save2='cases.txt'):
  f = open(path['post_path'])
  posts = f.readlines()
  f.close()
  f = open(path['target_path'])
  targets = f.readlines()
  f.close()
  results = dict()
  for ele in model_result:
    f = open(ele['result_path'])
    results[ele['model']] = f.readlines()
    f.close()
  f = open(save2,'w')
  for i in range(len(posts)):
    f.write('q:'+posts[i])
    f.write('target:'+targets[i])
    for k in results.keys():
      f.write(k+':'+results[k][i])
    f.write('\n')
  f.close()

if __name__=='__main__':
  run()
