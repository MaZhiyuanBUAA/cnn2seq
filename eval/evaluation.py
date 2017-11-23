#coding:utf-8

import sys
import random
import numpy as np
import math
import pickle
from config import path,distinct_config
from collections import defaultdict

#p = sys.argv[1]
#f = open(p)
#true_sents = f.readlines()
#f.close()

def distinct1(sents,test_size,test_times):
  values = []
  for i in range(test_times):
    sampled = random.sample(sents,test_size)
    vocab = {}
    for ele in sampled:
      words = ele.split()
      for w in words:
        try:
          vocab[w] += 1
        except:
          vocab[w] = 1
    values.append((len(vocab),sum(vocab.values())))
  return (test_size,np.mean([ele[0] for ele in values]),
          np.mean([ele[1] for ele in values]),
          np.mean([1.*ele[0]/ele[1] for ele in values]))

def mean_length(sents):
  num = len(sents)
  total_words = 0.
  for ele in sents:
    total_words += len(ele.split())
  return total_words/num

def build_idfTable(fpath,save2='idfTable.pkl'):
  f = open(fpath)
  s = f.readline()
  vocab = defaultdict(int)
  D = 0
  while s:
    words = s.split()
    for w in list(set(words)):
      vocab[w] += 1
    D += 1
    if D%100000 == 0:
      print('processing line %d'%D)
    s = f.readline()
  f.close()
  idfTable = dict([(k,math.log(D/(v+1.),10)) for k,v in vocab.items()])
  f = open(save2,'w')
  pickle.dump(idfTable,f)
  f.close()

def mean_max_idf(sents,idf_table):
  values = []
  for ele in sents:
    v = []
    for w in ele.split():
      try:
        v.append(idf_table[w])
      except:
        pass
    if len(v)==0:
      values.append(0)
      continue
    values.append(np.max(v))
  if len(values)==0:
    return 0
  return np.mean(values)

def single_embeddingBasedEvaluation(predict,target,embeddings,vocab):
  #print(vocab.items()[:10])
  #print(embeddings.shape)
  #print(embeddings[3])
  predict = [vocab[ele] for ele in predict.split()]
  predict = [embeddings[ele] for ele in predict]
  tmp = []
  for ele in target.split():
    try:
      tmp.append(vocab[ele])
    except:
      tmp.append(3)
  target = [embeddings[ele] for ele in tmp]
  def cosine_similarity(v1,v2):
    #print(v1.shape,v2.shape)
    #assert v1.shape[0]==1
    try:
      return np.sum(v1*v2,axis=1)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2,axis=1))+1e-10)
    except:
      return np.sum(v1*v2)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2))+1e-10)
  Gpt = 0
  emb_t = np.array(target)
  #print(emb_t.shape)
  for ele in predict:
    Gpt += np.max(cosine_similarity(ele,emb_t))
  Gtp = 0
  emb_p = np.array(predict)
  for ele in target:
    Gtp += np.max(cosine_similarity(ele,emb_p))
  GM =  0.5*(Gpt/len(predict)+Gtp/len(target))
  #calculate EA
  tmp_p = np.sum(emb_p,axis=0)
  tmp_t = np.sum(emb_t,axis=0)
  EA = cosine_similarity(tmp_p/np.sqrt(np.sum(tmp_p**2)),tmp_t/np.sqrt(np.sum(tmp_t**2)))
  #calculate Vector Extrema of predict
  max_ = np.max(emb_p,axis=0)
  min_ = np.min(emb_p,axis=0)
  abs_min_ = np.abs(min_)
  VE_p = np.sign(max_+1e-10-abs_min_)*np.max([max_,abs_min_],axis=0)
  #calculate VE of target
  max_ = np.max(emb_t,axis=0)
  min_ = np.min(emb_t,axis=0)
  abs_min_ = np.abs(min_)
  #print(emb_t)
  #print(np.sign(max_-abs_min_))
  VE_t = np.sign(max_+1e-10-abs_min_)*np.max([max_,abs_min_],axis=0)
  VE = cosine_similarity(VE_p,VE_t)
  #print(VE_t,VE_p)
  #print(GM,EV,VE)
  return GM,EA,VE
def embeddingBasedEvaluation(predicts,targets,emb,vocab):
  lp, lt = len(predicts),len(targets)
  if not (lp==lt):
    return ValueError,'number of predicts %d not equal to number of targets %d'%(lp,lt)
  GM,EA,VE = 0,0,0
  #print(emb[0],emb[1],emb[2])
  for ind in range(lp):
    dGM,dEA,dVE = single_embeddingBasedEvaluation(predicts[ind],targets[ind],emb,vocab)
    GM += dGM
    EA += dEA
    VE += dVE
  return GM/lp,EA/lp,VE/lp
def read():
  try:
    f = open(path['pred_path'])
    predicts = f.readlines()
    f.close()
  except:
    predicts = None
  try:
    f = open(path['target_path'])
    targets = f.readlines()
    f.close()
  except:
    targets = None
  try:
    f = open(path['emb_path'])
    emb = pickle.load(f)
    f.close()
  except:
    emb = None
  try:
    f = open(path['vocab_path'])
    vocab = f.readlines()
    f.close()
    vocab = dict([(ele.strip(),ind) for ind,ele in enumerate(vocab)])
  except:
    vocab = None
  try:
    f = open(path['idf_path'])
    idfTable = pickle.load(f)
    f.close()
  except:
    idfTable = None
  return predicts,targets,emb,vocab,idfTable
#size = 100
#s_num = 1000
#print(distinct1(true_sents,size,s_num))
if __name__=='__main__':
  #fpath = '/home/zyma/work/source.txt'
  #build_idfTable(fpath)
  predicts,targets,emb,vocab,idfTable = read()
  #print('mean_length:%f'%mean_length(predicts))
  for ele in distinct_config:
    print('test_size:%d,mean_num_dictinct1:%d,mean_total_word:%d,distinct1:%f'%(distinct1(predicts,ele['test_size'],ele['test_times'])))
  #print('mean_max_idf:%f'%mean_max_idf(predicts,idfTable))
  print('GM=%f,EA=%f,VE=%f'%embeddingBasedEvaluation(predicts,targets,emb,vocab))

