#coding:utf-8

import tensorflow as tf
from model_utils import create_cnn2seq,create_seq2seq,create_rethink
from config import cnn2seq_config,seq2seq_config,rethink_config
import data_utils
from data_utils import read_data_lm,read_data
import time,os
import math
import sys

def train_LM():
    print("Preparing dialog data in %s" % lm_config.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(lm_config.data_dir, lm_config.vocab_size)

    with tf.Session() as sess:

        # Create model.
        print("Creating %d layers of %d units." % (lm_config.num_layers, lm_config.size))
        #model = create_LM(sess)
        model = create_LM(sess)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % lm_config.max_train_data_size)
        dev_set = read_data_lm(dev_data)
        train_set = read_data_lm(train_data, lm_config.max_train_data_size)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        max_global_step = 120000
        while model.global_step.eval()<max_global_step:
          # Get a batch and make a step.
          start_time = time.time()
          inputs,weights = model.get_batch(train_set)
          time_ = time.time()-start_time
          #print(time_)
          step_loss,_ = model.step(sess, inputs,weights)
          #print(time.time()-start_time-time_)
          step_time += (time.time() - start_time) / lm_config.steps_per_checkpoint
          loss += step_loss / lm_config.steps_per_checkpoint
          current_step += 1
          #print('loss:%f'%loss)
          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % lm_config.steps_per_checkpoint == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity %.2f" %
                   (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))

            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)

            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(lm_config.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0,0.0
            sys.stdout.flush()

def test(create_func,conf):
    print("Preparing dialog data in %s" % conf.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(conf.data_dir, conf.vocab_size)

    with open(conf.model_dir+'/results.txt','a') as f:
        sess = tf.Session()
        # Create model.
        print("Creating %d layers of %d units." % (conf.num_layers, conf.size))
        #model = create_LM(sess)
        model = create_func(sess,conf)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % conf.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data,conf.max_train_data_size)
        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later

        # This is the training loop.
        i = 0
        eval_ppl = 0
        max_global_step = 120000
        while (i+1)*model.batch_size<len(dev_set):
            #eval loss
            input_data = model.get_batch(dev_set[i*model.batch_size:(i+1)*model.batch_size],random_choice=False)
            eval_loss,_ = model.step(sess,*input_data,forward_only=True)
            eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            print("current:%d eval: perplexity %.2f"%(i,eval_ppx))
            eval_ppl += eval_ppx
            #print('mean eval ppl:%f'%(eval_ppl/100))
            i += 1
            sys.stdout.flush()
        print('mean eval ppl:%f'%(eval_ppl/(i)))
if __name__ == '__main__':
  arg = sys.argv[1]
  if arg == 'cnn2seq':
    test(create_cnn2seq,cnn2seq_config())
  if arg == 'seq2seq':
    test(create_seq2seq,seq2seq_config())  
  if arg == 'rethink':
    test(create_rethink,rethink_config())
  

