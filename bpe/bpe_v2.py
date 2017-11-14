import re, collections
import time
import sys

def initialize_pairs(vocab):
  pairs = collections.defaultdict(int)
  all_pairs = collections.defaultdict(int)
  for word,freq in vocab.items():
    symbols = word.split()
    l = len(symbols)
    for i in range(l-1):
      pairs[symbols[i],symbols[i+1]] += freq
      all_pairs[symbols[i],symbols[i+1]] += freq
      if i+2<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2]] += freq
        #pairs[symbols[i],symbols[i+1]+symbols[i+2]] += freq
      if i+3<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2],symbols[i+3]] += freq
        #pairs[symbols[i],symbols[i+1]+symbols[i+2]+symbols[i+3]] += freq
      if i+4<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2],symbols[i+3],symbols[i+4]] += freq
        #pairs[symbols[i],symbols[i+1]+symbols[i+2]+symbols[i+3]+symbols[i+4]] += freq
      if i+5<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2],symbols[i+3],symbols[i+4],symbols[i+5]] += freq
        #pairs[symbols[i],symbols[i+1]+symbols[i+2]+symbols[i+3]+symbols[i+4]+symbols[i+5]] += freq
      if i+6<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2],symbols[i+3],symbols[i+4],symbols[i+5],symbols[i+6]] += freq
      if i+7<l:
        all_pairs[symbols[i],symbols[i+1],symbols[i+2],symbols[i+3],symbols[i+4],symbols[i+5],symbols[i+6],symbols[i+7]] += freq
  return pairs,all_pairs

def update_pairs(pairs,all_pairs,best):
  st = time.time()
  del pairs[best]
  #new_pairs = collections.defaultdict(int)
  for k in pairs.keys():
    k1,k2 = k
    if k2 == best[0]:
      pairs[k] -= all_pairs[tuple(k1+k2+best[1])]
      pairs[k1,k2+best[1]] = all_pairs[tuple(k1+k2+best[1])]
      #print(k,k1,k2+best[1],best)
    if k1 == best[1]:
      pairs[k] -= all_pairs[tuple(best[0]+k1+k2)]
      pairs[best[0]+k1,k2] = all_pairs[tuple(best[0]+k1+k2)]
      #print(k,best[0]+k1,k2,best)
  for k in pairs.keys():
    if pairs[k]<=0:
      del pairs[k]
  return pairs,time.time()-st

def get_pairs(vocab):
  pairs = collections.defaultdict(int)
  for word, freq in vocab.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i],symbols[i+1]] += freq
  return pairs

def merge_vocab(pair, v_in):
  st = time.time()
  v_out = {}
  bigram = re.escape(' '.join(pair))
  p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
  for word in v_in:
    w_out = p.sub(''.join(pair), word)
    v_out[w_out] = v_in[word]
  return v_out,time.time()-st

def initialize_vocab():
  f = open('data/test.in')
  s = f.readline()
  v = collections.defaultdict(int)
  while s:
    words = s.decode('utf-8').split()
    for w in words:
      v[w] += 1
    s = f.readline()
  return v
def main():
  num_merges = 3000
  start_time = time.time()
  #vocab_init = {'l o w *' : 5, 'l o w e r *' : 2,'n e w e s t *':6, 'w i d e s t *':3}
  vocab_init = initialize_vocab()
  print('vocab_size:%d'%len(vocab_init.keys()))
  vocab_item = vocab_init.items()
  vocab_item = sorted(vocab_item,key=lambda x:x[1],reverse=True)
  print('sorted ....')
  vocab = dict(vocab_item[:num_merges])
  vocab_bpe = dict([(' '.join(k),v) for k,v in vocab_item[num_merges:]])
  #vocab_bpe = dict(vocab_item[30000:])
  print('start merge ...')
  f = open('vocab_init_test.txt','w')
  f.write('\n'.join(vocab.keys()).encode('utf-8'))
  f.close()
  #pairs = get_stats_init(vocab_bpe)
  #vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,'n e w e s t </w>':6, 'w i d e s t </w>':3}
  #num_merges = 3000
  mst = time.time()
  pairs,all_pairs = initialize_pairs(vocab_bpe)
  #vocab_bpe_=vocab_bpe
  #pairs_ = get_pairs(vocab_bpe_)
  #print(all_pairs)
  #print(pairs)
  #pairs_items = [(k,v) for k,v in pairs.items()]
  #pairs_items = sorted(pairs_items,key=lambda x:x[1],reverse=True)
  #print(pairs_items)
  #print(pairs)
  #print(pairs_)
  #print('******************************')
  time_0,time_1 = 0,0
  for i in range(num_merges):
    i += 1
    if i%100==0:
      print('merge times:%d,time per merge:%f,merge_time:%f,update_time:%f'%(i,(time.time()-mst)/(i),time_0/i,time_1/i))
    #print(pairs)
    best = max(pairs,key=pairs.get)
    #print best
    #best_ = max(pairs_,key=pairs_.get)
    if pairs[best]<=1:
      print('early stop in merge=%d'%i)
      break
    #print(best)
    vocab_bpe,dt0 = merge_vocab(best, vocab_bpe)
    #vocab_bpe_ = merge_vocab(best_,vocab_bpe_)
    pairs,dt1 = update_pairs(pairs,all_pairs,best)
    time_0 += dt0
    time_1 += dt1
    if not pairs:
      print('early stop because of vide pairs, merge=%d'%i)
      break
    #pairs_ = get_pairs(vocab_bpe_)
    #print(best,best_)
    #print(pairs)
    #print(pairs_)
    #print('****************************')
    #print(vocab_bpe)
    #print(vocab)
    sys.stdout.flush()
  f = open('vocab_bpe_test.txt','w')
  f.write('\n'.join(vocab_bpe.keys()).encode('utf-8'))
  f.close()
  print('time used:%f'%(time.time()-start_time))

def data_process_useBPE():
  data_f = open('data/test1k.resp')
  vocab_f = open('vocab_luntan.bpe')
  vocab_bpe = vocab_f.readlines()
  vocab_bpe = [ele.strip().decode('utf-8') for ele in vocab_bpe]
  vocab_bpe = dict([(ele.replace(' ',''),ele) for ele in vocab_bpe])
  print(vocab_bpe.items()[:10])
  dataByBpe_f = open('data_bpe/test1k.resp','w')
  s = data_f.readline()
  i = 1
  while s:
    tmp = s.decode('utf-8').split()
    tmp_ = []
    for w in tmp:
      try:
        tmp_.append(vocab_bpe[w])
        #print('%s --> %s'%(w,vocab_bpe[w]))
      except:
        tmp_.append(w)
    dataByBpe_f.write(' '.join(tmp_).encode('utf-8')+'\n')
    s = data_f.readline()
    i += 1
  data_f.close()
  dataByBpe_f.close()

if __name__ == '__main__':
  data_process_useBPE()
  #main()
