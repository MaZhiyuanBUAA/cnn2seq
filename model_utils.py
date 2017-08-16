#coding:utf-8

import tensorflow as tf
from model import cnn2seq
from config import dm_config 
dm_config = dm_config()


def create_cnn2seq(session):
  model = cnn2seq(dm_config.input_len,
                            dm_config.output_len,
                            dm_config.size,
                            dm_config.num_layers,
                            dm_config.learning_rate,
                            dm_config.decay_factor,
                            dm_config.batch_size,
                            dm_config.max_gradient_norm,
                            dm_config.new_embeddings,
                            dm_config.vocab_size,
                            dm_config.emb_dim,
                            dm_config.num_heads,
                            name='seq2seq_lm')
  ckpt = tf.train.get_checkpoint_state(dm_config.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model
if __name__ == '__main__':
  with tf.Session() as sess:
    model=create_cnn2seq(sess)
