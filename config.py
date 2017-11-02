#coding:utf-8
TEST_DATASET_PATH = 'data/test1k.query'

class lm_config(object):
  def __init__(self):
    self.data_dir = '/home/zyma/work/data_lm'
    self.model_dir = 'l_models'
    self.vocab_size = 110000
    self.max_train_data_size = 1000000
 
    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.num_steps = 20
    
    self.max_gradient_norm = 5.0 
    self.learning_rate = 0.05
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100


class cnn2seq_config(object):
  def __init__(self):
    self.data_dir = 'data_bpe'
    self.model_dir = 'cnn2seq_models'
    self.vocab_size = 60000
    self.max_train_data_size = 10000000
    self.results_dir = 'results'
    
    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.input_len = 15
    self.output_len = 20
    self.new_embeddings = False
    self.num_heads = 1
 
    self.max_gradient_norm = 5.0
    self.learning_rate = 0.5
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100

class seq2seq_config(object):
  def __init__(self):
    self.data_dir = 'data_bpe'
    self.model_dir = 'seq2seq_models'
    self.vocab_size = 100000
    self.max_train_data_size = 10000000
    self.results_dir = 'results/results_4layers.txt'
    
    self.num_layers = 4
    self.size = 128
    self.emb_dim = 128
    self.input_len = 15
    self.output_len = 20
    self.new_embeddings = False
    self.num_heads = 1
 
    self.max_gradient_norm = 5.0
    self.learning_rate = 0.5
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100

class rethink_config(object):
  def __init__(self):
    self.data_dir = '/home/zyma/work/data_daily_punct'
    self.model_dir = 'rethink_models/old_model'
    self.vocab_size = 110000
    self.max_train_data_size = 1000000
    self.results_dir = 'results'

    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.input_len = 15
    self.output_len = 20
    self.new_embeddings = False
    self.num_heads = 1

    self.max_gradient_norm = 5.0
    self.learning_rate = 0.5
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100
