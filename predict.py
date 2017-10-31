import os

import tensorflow as tf
import data_utils
from config import TEST_DATASET_PATH, cnn2seq_config,seq2seq_config,rethink_config
from model_utils import create_seq2seq,create_cnn2seq,create_rethink,get_predicted_sentence



def predict(create_func,conf):
    def _get_test_dataset():
        with open(TEST_DATASET_PATH) as test_fh:
            test_sentences = [s.strip() for s in test_fh.readlines()]
        return test_sentences

    results_filename = '_'.join(['results_cnn2seq', str(conf.num_layers), str(conf.size), str(conf.vocab_size)])
    results_path = os.path.join(conf.results_dir, results_filename)
    conf.batch_size = 1
    #with tf.Session() as sess, open(results_path, 'w') as results_fh:
    with tf.Session() as sess,open(results_path,'w') as results_fh:
        # Create model and load parameters.
        model = create_func(sess,conf,forward_only=True)
        #model.batch_size = 100  # We decode one sentence at a time.

        # Load vocabularies.
        vocab_path = os.path.join(conf.data_dir, "vocab%d.in" % conf.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        print(vocab.items()[:20])

        test_dataset = _get_test_dataset()
        #test_dataset = test_dataset[374:]
        #predicted_sentences = beam_search(test_dataset,vocab,rev_vocab,model,sess)
        #results_fh.write('\n'.join(predicted_sentences))

        for sentence in test_dataset:
            # Get token-ids for the input sentence.
            #best,predicted_sentences,scores = beam_search(sentence, vocab, rev_vocab, model, sess)
            predicted_sentence = get_predicted_sentence(sentence, vocab, rev_vocab, model, sess)
            print (sentence+' -> '+predicted_sentence)
            #print ('\n'.join([str(ele)+','+predicted_sentences[ind] for ind,ele in enumerate(scores)]))
            #print(len(scores))
            #results_fh.write(best+'\n')
            results_fh.write(predicted_sentence+'\n')

if __name__=='__main__':
  predict(create_rethink,rethink_config())
