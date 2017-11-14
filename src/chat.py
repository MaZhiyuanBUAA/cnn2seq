import os
import sys
import jieba
import tensorflow as tf
import data_utils
from config import TEST_DATASET_PATH, cnn2seq_config,seq2seq_config,rethink_config
from model_utils import create_seq2seq,create_cnn2seq,create_rethink,get_predicted_sentence



def predict(create_func,conf):

    results_filename = '_'.join(['results_cnn2seq', str(conf.num_layers), str(conf.size), str(conf.vocab_size)])
    results_path = os.path.join(conf.results_dir, results_filename)
    conf.batch_size = 1
    #with tf.Session() as sess, open(results_path, 'w') as results_fh:
    with tf.Session() as sess:
        # Create model and load parameters.
        model = create_func(sess,conf,forward_only=True)

        vocab_path = os.path.join(conf.data_dir, "vocab%d.in" % conf.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        sentence = raw_input('>')
        while sentence:

            sentence = ' '.join(jieba.cut(sentence)).encode('utf-8')
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
            p = predicted_sentence.split()
            p1 = p[:model.output_len]
            p2 = p[model.output_len:]
            for ind, ele in enumerate(p1):
              if ele == "_EOS":
                print(' '.join(p1[:ind]))
                break
            for ind, ele in enumerate(p2):
              if ele == "_EOS":
                print(" ".join(p2[:ind]))
                break
            #print (" ".join(p1]))
            sentence = raw_input('>')

if __name__=='__main__':
  arg = sys.argv[1]
  if arg=='rethink':
    predict(create_rethink,rethink_config())
  if arg=='seq2seq':
    predict(create_seq2seq,seq2seq_config())
  if arg=='cnn2seq':
    predict(create_cnn2seq,cnn2seq_config())
