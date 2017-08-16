#coding:utf-8

class lm_config(object):
  def __init__(self):
    self.data_dir = '/home/zyma/work/data_lm'
    self.model_dir = 'l_models'
    self.vocab_size = 110000
    self.max_train_data_size = 100000
 
    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.num_steps = 20
    
    self.max_gradient_norm = 5.0 
    self.learning_rate = 0.05
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100


class dm_config(object):
  def __init__(self):
    self.data_dir = '/home/zyma/work/data_lm'
    self.model_dir = 'd_models'
    self.vocab_size = 110000
    self.max_train_data_size = 100000

    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.input_len = 15
    self.output_len = 20
    self.new_embeddings = False
    self.num_heads = 1
 
    self.max_gradient_norm = 5.0
    self.learning_rate = 0.05
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100

class dmc_config(object):
  def __init__(self):
    self.data_dir = '/home/zyma/work/data_lm'
    self.model_dir = 'dmc_models'
    self.vocab_size = 110000
    self.max_train_data_size = 100000

    self.num_layers = 3
    self.size = 128
    self.emb_dim = 128
    self.input_len = 15
    self.output_len = 20
    self.num_heads = 1

    self.max_gradient_norm = 5.0
    self.learning_rate = 0.05
    self.decay_factor = 0.99
    self.batch_size = 128
    self.steps_per_checkpoint = 100

