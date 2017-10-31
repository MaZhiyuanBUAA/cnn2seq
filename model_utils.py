#coding:utf-8

import tensorflow as tf
from model import cnn2seq,seq2seq,cnn2seq_rethink
from config import cnn2seq_config,seq2seq_config,rethink_config
import data_utils
import numpy as np

def create_cnn2seq(session,conf,forward_only=False):
  model = cnn2seq(conf.input_len,
                            conf.output_len,
                            conf.size,
                            conf.num_layers,
                            conf.learning_rate,
                            conf.decay_factor,
                            conf.batch_size,
                            conf.max_gradient_norm,
                            conf.new_embeddings,
                            conf.vocab_size,
                            conf.emb_dim,
                            conf.num_heads,
                            forward_only = forward_only,
                            name='cnn2seq')
  ckpt = tf.train.get_checkpoint_state(conf.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model
def create_seq2seq(session,conf,forward_only=False):
  model = seq2seq(conf.input_len,
                            conf.output_len,
                            conf.size,
                            conf.num_layers,
                            conf.learning_rate,
                            conf.decay_factor,
                            conf.batch_size,
                            conf.max_gradient_norm,
                            conf.new_embeddings,
                            conf.vocab_size,
                            conf.emb_dim,
                            conf.num_heads,
                            forward_only = forward_only,
                            name='seq2seq_lm')
  ckpt = tf.train.get_checkpoint_state(conf.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model
def create_rethink(session,conf,forward_only=False):
  model = cnn2seq_rethink(conf.input_len,
                            conf.output_len,
                            conf.size,
                            conf.num_layers,
                            conf.learning_rate,
                            conf.decay_factor,
                            conf.batch_size,
                            conf.max_gradient_norm,
                            conf.new_embeddings,
                            conf.vocab_size,
                            conf.emb_dim,
                            conf.num_heads,
                            forward_only = forward_only,
                            name='seq2seq_lm')
  ckpt = tf.train.get_checkpoint_state(conf.model_dir)
  if ckpt:
    print("Reading seq2seq Embedded LM from %s"%ckpt.model_checkpoint_path)
    model.saver.restore(session,ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh params.")
    session.run(tf.global_variables_initializer())
    #session.run(model.assign_params)
  return model

def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    print(input_token_ids)
    # Which bucket does it belong to?
    outputs = [data_utils.GO_ID]
    feed_data = [(input_token_ids, outputs)]
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data)
    # Get output logits for the sentence.
    _,output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, forward_only=True)
    outputs = []
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    for logit in output_logits:
        selected_token_id = int(np.argmax(logit, axis=1))
        print((selected_token_id,np.max(logit)))
        outputs.append(selected_token_id)
        #if selected_token_id == data_utils.EOS_ID:
        #        break
        #else:
        #        outputs.append(selected_token_id)
    # Forming output sentence on natural language
    outputs = ' '.join([rev_vocab[i] for i in outputs])
    return outputs

if __name__ == '__main__':
  with tf.Session() as sess:
    model=create_cnn2seq(sess,seq2seq_config())
