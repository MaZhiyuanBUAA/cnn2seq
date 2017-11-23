#coding:utf-8

import csv
import random
import sys
import json

reload(sys)
sys.setdefaultencoding('utf-8')

f = open('../data_bpe/test1k.query')
queries = f.readlines()
queries = [ele.decode('utf-8').encode('gbk') for ele in queries]
f.close()

f = open('../results/seq2seq_4layers.txt')
resp0s = f.readlines()
resp0s = [ele.decode('utf-8').encode('gbk') for ele in resp0s]
f.close()

f = open('../results/inputs_cnn.txt')
resp1s = f.readlines()
resp1s = [ele.decode('utf-8').encode('gbk') for ele in resp1s]
f.close()

csvf = open('results.csv','w')
writer = csv.writer(csvf)

logs = []
for ind,ele in enumerate(queries):
  e = random.random()
  logs.append(e)
  if e>0.5:
    resp0 = resp0s[ind]
    resp1 = resp1s[ind]
  else:
    resp0 = resp1s[ind]
    resp1 = resp0s[ind]
  writer.writerow((ele.replace(' ',''),resp0.replace(' ','')))
  writer.writerow(('',resp1.replace(' ','')))
  writer.writerow(('',''))
writer.writerow(('token',json.dumps(logs)))
csvf.close()
