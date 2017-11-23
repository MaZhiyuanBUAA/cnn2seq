#coding:utf-8
import copy
import tensorflow as tf
import random
import data_utils
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops

linear = core_rnn_cell_impl._linear
def _extract_argmax_and_embed(embedding,output_projection=None,update_embedding=True):

  def loop_function(prev, _):
    if output_projection is not None:
      prev = tf.nn.xw_plus_b(prev, output_projection[0], output_projection[1])
    prev_symbol = math_ops.argmax(prev, 1)
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev

  return loop_function

class cnn2seq(object):
  def __init__(self,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               new_embeddings=False,vocab_size=None,emb_dim=None,
               num_heads=1,num_samples=512,forward_only=False,
               with_rnn=True,position='INPUTS',Filter='MULTI',name='cnn2seq'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.emb_dim = emb_dim
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[self.batch_size],name='input{}'.format(i)) for i in range(input_len)]
      self.decoder_inputs = [tf.placeholder(tf.int32,shape=[self.batch_size],name='decoder_input{}'.format(i)) for i in range(output_len)]
      self.targets = self.decoder_inputs[1:]+[tf.placeholder(tf.int32,shape=[self.batch_size],name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,shape=[self.batch_size],name='target_weight{}'.format(i)) for i in range(output_len)]
      self.mask = [tf.placeholder(tf.float32,shape=[self.batch_size,emb_dim],name='mask{}'.format(i)) for i in range(input_len)]
      self. embeddings = tf.get_variable(name='embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer())

      emb_inps = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.inputs]
      emb_decoder_inps = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.decoder_inputs]
      if with_rnn == True:
        with tf.variable_scope('rnn_encoder'):
          encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
          encoder_outputs, state = tf.contrib.rnn.static_rnn(encoder_cell,emb_inps,dtype=tf.float32)
        #print('state shape',state.get_shape())
      #cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      #encoder_outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inps,dtype=tf.float32)
      if with_rnn == True:
        top_states = [tf.reshape(ele,[-1,1,size]) for ele in encoder_outputs]
      else:
        top_states = [tf.reshape(ele,[-1,1,size]) for ele in emb_inps]
      #top_states = [tf.reshape(ele,[-1,1,size]) for ele in emb_inps]
      mask_ = [tf.reshape(ele,[-1,1,size]) for ele in self.mask]
      att_states = tf.concat(top_states,axis=1)
      mask_ = tf.concat(mask_,axis=1)
      init_common_states = tf.expand_dims(att_states,-1)
      if with_rnn==True:
        if position=='INPUTS':
          cnn_states = [tf.reshape(ele,[-1,1,size]) for ele in emb_inps]
        if position=='STATES':
          cnn_states = top_states
        init_common_states = tf.expand_dims(tf.concat(cnn_states,axis=1),-1)
      mask_ = tf.expand_dims(mask_,-1)

      def cnn(n_layer,inputs):
          if Filter == 'MULTI':
            channels = size/4
            filter_8 = tf.get_variable('filter_8',[8,size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
            filter_5 = tf.get_variable('filter_5',[5,size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
            filter_3 = tf.get_variable('filter_3',[3,size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
            filter_1 = tf.get_variable('filter_1',[1,size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))

            b8 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b8")
            b5 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b5")
            b3 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b3")
            b1 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b1")

            #print(inputs.get_shape())
            #filter1 = tf.get_variable('filter1',[3,3,1,2])
            #filter2 = tf.get_variable('filter2',[3,3,2,4])
            inputs_8 = tf.nn.conv2d(inputs,filter_8,[1,1,1,1],padding="VALID")
            inputs_5 = tf.nn.conv2d(inputs,filter_5,[1,1,1,1],padding="VALID")
            inputs_3 = tf.nn.conv2d(inputs,filter_3,[1,1,1,1],padding="VALID")
            inputs_1 = tf.nn.conv2d(inputs,filter_1,[1,1,1,1],padding="VALID")
            #print(inputs_8.get_shape())
            #print(inputs_5.get_shape())
            #print(inputs_3.get_shape())
            #print(inputs_1.get_shape())


            inputs_8 = tf.nn.tanh(tf.nn.bias_add(inputs_8,b8))
            inputs_5 = tf.nn.tanh(tf.nn.bias_add(inputs_5,b5))
            inputs_3 = tf.nn.tanh(tf.nn.bias_add(inputs_3,b3))
            inputs_1 = tf.nn.tanh(tf.nn.bias_add(inputs_1,b1))

            inputs_8 = tf.nn.max_pool(inputs_8,[1,8,1,1],[1,1,1,1],padding="VALID")
            inputs_5 = tf.nn.max_pool(inputs_5,[1,11,1,1],[1,1,1,1],padding="VALID")
            inputs_3 = tf.nn.max_pool(inputs_3,[1,13,1,1],[1,1,1,1],padding="VALID")
            inputs_1 = tf.nn.max_pool(inputs_1,[1,15,1,1],[1,1,1,1],padding="VALID")
            #print(inputs_1.get_shape())

            outputs = tf.concat([inputs_8,inputs_5,inputs_3,inputs_1],axis=1)
          else:
            assert isinstance(Filter,int)
            channels = size
            filt = tf.get_variable('filt',[Filter,size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
            b = tf.Variable(tf.constant(0.1, shape=[channels]), name="b")
            inputs_ = tf.nn.conv2d(inputs,filt,[1,1,1,1],padding="VALID")
            inputs_ = tf.nn.max_pool(inputs_,[1,Filter,1,1],[1,1,1,1],padding="VALID")
            outputs = tf.nn.tanh(tf.nn.bias_add(inputs_,b))

          
          outputs = tf.reshape(outputs,[self.batch_size,-1])
          
          #wo = tf.get_variable('wo',[4*size,size],initializer=tf.truncated_normal_initializer(stddev=0.1))
          #bo = tf.get_variable('bo',[size],initializer=tf.truncated_normal_initializer(stddev=0.1))

          print(outputs.get_shape())
          #inputs = tf.nn.max_pool(inputs,[1,self.input_len,1,1],[1,1,1,1],padding="VALID")
          #inputs = tf.reshape(inputs,[self.batch_size,-1])
          #return tf.nn.xw_plus_b(outputs,wo,bo)
          return outputs
      with tf.variable_scope('cnn_encoder'):
          common_states = cnn(2,mask_*init_common_states)
          print(common_states.get_shape())

      att_length = att_states.get_shape()[1].value
      att_size = att_states.get_shape()[2].value
      att_vec_size = att_size

      hidden = tf.reshape(att_states,[-1,att_length,1,att_size])
      hidden_features = []
      v = []
      for a in range(num_heads):
        k = tf.get_variable('AttnW_%d'%a,[1,1,att_size,att_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden,k,[1,1,1,1],'SAME'))
        v.append(tf.get_variable('AttnV_%d'%a,[att_vec_size]))

      def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        #if tf.contrib.framework.nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #  query_list = tf.contrib.framework.nest.flatten(query)
        #  for q in query_list:  # Check that ndims == 2 if specified.
        #    ndims = q.get_shape().ndims
        #    if ndims:
        #      assert ndims == 2
        #  query = tf.concat(query_list, 1)
        for a in xrange(num_heads):
          with tf.variable_scope("Attention_%d" % a):
            y = linear(query, att_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, att_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                  [2, 3])
            a_ = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a_, [-1, att_length, 1, 1]) * hidden, [1, 2])
            ds.append(tf.reshape(d, [-1, att_size]))
        return ds
      proj_w = tf.get_variable('proj_w',[emb_dim,vocab_size])
      proj_b = tf.get_variable('proj_b',[vocab_size])
      w_t = tf.transpose(proj_w)
      loop_function = _extract_argmax_and_embed(self.embeddings,output_projection=[proj_w,proj_b])      
      with tf.variable_scope('loop'):
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        if with_rnn == False:
          state = decoder_cell.zero_state(self.batch_size,dtype=tf.float32)
        #print('state shape:',state.get_shape())
        #state = tf.zeros([self.batch_size,size])
        attns = attention(state)
        #with tf.variable_scope('first_inp'):
        #  inp = linear(attns,emb_dim,True)
        outputs = []
        prev = None
        for i,inp in enumerate(emb_decoder_inps):
          if i>0:
            variable_scope.get_variable_scope().reuse_variables()
          if forward_only and prev is not None:
            inp = loop_function(prev,i)
          #print(inp.get_shape(),attns[0].get_shape())
          x = linear([common_states]+[inp]+attns,emb_dim,True)
          cell_output,state = decoder_cell(x,state)
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                             reuse=True):
            attns = attention(state)
          with tf.variable_scope('AttnOutputProjection'):
            output = linear([cell_output]+attns,emb_dim,True)
            prev = output
          outputs.append(output)
      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(w_t,proj_b,labels,inputs,num_samples,vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,proj_w,proj_b) for ele in outputs]
    self.params = []
    #rnn_encoder_params = []
    #cnn_encoder_params = []
    
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        self.params.append(ele)
    #assign params of LM
    #self.assign_params = [tf.assign(ele,LM.params[ind+2]) for ind, ele in enumerate(self.params_use_LM)]
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params)
    #print(gradients)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))
    #tmp = [ele.name for ele in self.params_use_LM]
    #print(len(tmp))
    #print('\n'.join(tmp))

  def step(self,sess,encoder_inps,decoder_inps,weights,masks,forward_only=False):
    if len(encoder_inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    if len(weights)!=self.output_len:
      raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(decoder_inps)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = encoder_inps[i]
      input_feed[self.mask[i].name] = masks[i]
    for i in range(self.output_len):
      input_feed[self.target_weights[i].name] = weights[i]
      input_feed[self.decoder_inputs[i].name] = decoder_inps[i]

    input_feed[self.targets[-1].name] = np.zeros(self.batch_size)
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data, random_choice=True):
    encoder_inputs, decoder_inputs = [], []
    if not random_choice==True:
      assert len(data)==self.batch_size
    for ind in range(self.batch_size):
        if random_choice==True:
          encoder_input, decoder_input = random.choice(data)
        else:
          encoder_input, decoder_input = data[ind]
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_input)
        decoder_inputs.append(decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights,batch_masks = [], [], [],[]

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
      batch_mask = []
      for bathc_idx in range(self.batch_size):
         if encoder_inputs[batch_idx][length_idx] == data_utils.PAD_ID:
            batch_mask.append(np.zeros(self.emb_dim))
         else:
            batch_mask.append(np.ones(self.emb_dim))
      batch_masks.append(np.array(batch_mask))
      
    for length_idx in range(self.output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx<self.output_len-1:
          target = decoder_inputs[batch_idx][length_idx+1]

        if length_idx==self.output_len-1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs,batch_weights,batch_masks
      

class seq2seq(object):
  def __init__(self,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,
               new_embeddings=False,vocab_size=None,emb_dim=None,
               num_heads=1,num_samples=512,forward_only=False,name='seq2seq'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[None],name='input{}'.format(i)) for i in range(input_len)]
      self.decoder_inputs = [tf.placeholder(tf.int32,shape=[None],name='decoder_input{}'.format(i)) for i in range(output_len)]
      self.targets = self.decoder_inputs[1:]+[tf.placeholder(tf.int32,shape=[None],name='last_target')]
      self.target_weights = [tf.placeholder(tf.float32,shape=[None],name='target_weight{}'.format(i)) for i in range(output_len)]

      self.embeddings = tf.get_variable(name='embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32,initializer=tf.random_uniform_initializer())

      emb_inps = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.inputs]
      emb_decoder_inps = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.decoder_inputs]
      with tf.variable_scope('encoder_cell'):
        encoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        encoder_outputs,state = tf.contrib.rnn.static_rnn(encoder_cell,emb_inps,dtype=tf.float32)
      #encoder_outputs,_ = tf.contrib.rnn.static_rnn(cell,emb_inps,dtype=tf.float32)
      
      top_states = [tf.reshape(ele,[-1,1,encoder_cell.output_size]) for ele in encoder_outputs]
      att_states = tf.concat(top_states,axis=1)
      
      att_length = att_states.get_shape()[1].value
      att_size = att_states.get_shape()[2].value
      att_vec_size = att_size

      hidden = tf.reshape(att_states,[-1,att_length,1,att_size])
      hidden_features = []
      v = []
      for a in range(num_heads):
        k = tf.get_variable('AttnW_%d'%a,[1,1,att_size,att_vec_size])
        hidden_features.append(tf.nn.conv2d(hidden,k,[1,1,1,1],'SAME'))
        v.append(tf.get_variable('AttnV_%d'%a,[att_vec_size]))

      def attention(query):
        """Put attention masks on hidden using hidden_features and query."""
        ds = []  # Results of attention reads will be stored here.
        #if tf.contrib.framework.nest.is_sequence(query):  # If the query is a tuple, flatten it.
        #  query_list = tf.contrib.framework.nest.flatten(query)
        #  for q in query_list:  # Check that ndims == 2 if specified.
        #    ndims = q.get_shape().ndims
        #    if ndims:
        #      assert ndims == 2
        #  query = tf.concat(query_list, 1)
        for a in xrange(num_heads):
          with tf.variable_scope("Attention_%d" % a):
            y = linear(query, att_vec_size, True)
            y = tf.reshape(y, [-1, 1, 1, att_vec_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = tf.reduce_sum(v[a] * tf.tanh(hidden_features[a] + y),
                                  [2, 3])
            a_ = tf.nn.softmax(s)
            # Now calculate the attention-weighted vector d.
            d = tf.reduce_sum(
                tf.reshape(a_, [-1, att_length, 1, 1]) * hidden, [1, 2])
            ds.append(tf.reshape(d, [-1, att_size]))
        return ds
      proj_w = tf.get_variable('proj_w',[emb_dim,vocab_size])
      proj_b = tf.get_variable('proj_b',[vocab_size])
      w_t = tf.transpose(proj_w)
      loop_function = _extract_argmax_and_embed(self.embeddings,output_projection=[proj_w,proj_b])      
      with tf.variable_scope('loop'):
        decoder_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)])
        #print(state)
        #state = tf.zeros([self.batch_size,size])
        attns = attention(state)
        #with tf.variable_scope('first_inp'):
        #  inp = linear(attns,emb_dim,True)
        outputs = []
        prev = None
        for i,inp in enumerate(emb_decoder_inps):
          if i>0:
            variable_scope.get_variable_scope().reuse_variables()
          if forward_only and prev is not None:
            inp = loop_function(prev,i)
          x = linear([inp]+attns,emb_dim,True)
          cell_output,state = decoder_cell(x,state)
          with variable_scope.variable_scope(variable_scope.get_variable_scope(),
                                             reuse=True):
            attns = attention(state)
          with tf.variable_scope('AttnOutputProjection'):
            output = linear([cell_output]+attns,emb_dim,True)
            prev = output
          outputs.append(output)
      def nce_loss(labels,inputs):
        labels = tf.reshape(labels,[-1,1])
        return tf.nn.sampled_softmax_loss(w_t,proj_b,labels,inputs,num_samples,vocab_size)
      #self.losses = tf.reduce_mean(nce_loss(tf.concat(self.targets,axis=0),tf.concat(outputs,axis=0)))
      self.losses = sequence_loss(outputs,self.targets,self.target_weights,softmax_loss_function=nce_loss)
    self.predict = [tf.nn.xw_plus_b(ele,proj_w,proj_b) for ele in outputs]
    self.params = []
    self.params_use_LM = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        #if ele.name.find('use_LM')>-1:
        #  self.params_use_LM.append(ele)
        #else:
        self.params.append(ele)
    #assign params of LM
    #self.assign_params = [tf.assign(ele,LM.params[ind+2]) for ind, ele in enumerate(self.params_use_LM)]
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.losses,self.params)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params)
    tmp = [ele.name for ele in self.params]
    print(len(tmp))
    print('\n'.join(tmp))
    #tmp = [ele.name for ele in self.params_use_LM]
    #print(len(tmp))
    #print('\n'.join(tmp))

  def step(self,sess,encoder_inps,decoder_inps,weights,forward_only=False):
    if len(encoder_inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    if len(weights)!=self.output_len:
      raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(decoder_inps)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = encoder_inps[i]
    for i in range(self.output_len):
      input_feed[self.target_weights[i].name] = weights[i]
      input_feed[self.decoder_inputs[i].name] = decoder_inps[i]
    input_feed[self.targets[-1].name] = np.zeros(self.batch_size)
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if not forward_only:
       output_feed = [self.losses,self.update]
    else:
       output_feed = [self.losses,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data, random_choice=True):
    encoder_inputs, decoder_inputs = [], []
    if not random_choice==True:
      assert len(data)==self.batch_size
    for ind in range(self.batch_size):
        
        if random_choice==True:
          encoder_input, decoder_input = random.choice(data)
        else:
          encoder_input, decoder_input = data[ind]
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_input)
        decoder_inputs.append(decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)

    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))
    for length_idx in range(self.output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx<self.output_len-1:
          target = decoder_inputs[batch_idx][length_idx+1]

        if length_idx==self.output_len-1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs,batch_weights
     

 
class AS_model(object):
  def __init__(self,input_len,output_len,size,num_layers,
               learning_rate,decay_factor,batch_size,max_gradient_norm,vocab_size=None,emb_dim=None,
               name='answer_select'):
    with tf.variable_scope(name) as vs:
      self.input_len = input_len
      self.output_len = output_len
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate),trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate*decay_factor)
      self.global_step = tf.Variable(0,trainable=False)
      self.inputs = [tf.placeholder(tf.int32,shape=[self.batch_size],name='input{}'.format(i)) for i in range(input_len)]
      self.targets = [tf.placeholder(tf.int32,shape=[self.batch_size],name='output{}'.format(i)) for i in range(output_len)]
      self.labels = tf.placeholder(tf.int32,shape=[self.batch_size],name='labels')

      embeddings = tf.get_variable(name='embeddings',shape=[vocab_size,emb_dim],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))

      emb_inps = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.inputs]
      emb_targs = [tf.nn.embedding_lookup(embeddings,ele) for ele in self.targets]

      #cell_in = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      #cell_out = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(size) for i in range(num_layers)])
      def cnn(inputs,l):
        channels = size/4
        filter_8 = tf.get_variable('filter_8',[8,2*size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_5 = tf.get_variable('filter_5',[5,2*size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_3 = tf.get_variable('filter_3',[3,2*size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))
        filter_1 = tf.get_variable('filter_1',[1,2*size,1,channels],initializer=tf.truncated_normal_initializer(stddev=0.1))

        b8 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b8")
        b5 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b5")
        b3 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b3")
        b1 = tf.Variable(tf.constant(0.1, shape=[channels]), name="b1")

        inputs_8 = tf.nn.conv2d(inputs,filter_8,[1,1,1,1],padding="VALID")
        inputs_5 = tf.nn.conv2d(inputs,filter_5,[1,1,1,1],padding="VALID")
        inputs_3 = tf.nn.conv2d(inputs,filter_3,[1,1,1,1],padding="VALID")
        inputs_1 = tf.nn.conv2d(inputs,filter_1,[1,1,1,1],padding="VALID")

        inputs_8 = tf.nn.tanh(tf.nn.bias_add(inputs_8,b8))
        inputs_5 = tf.nn.tanh(tf.nn.bias_add(inputs_5,b5))
        inputs_3 = tf.nn.tanh(tf.nn.bias_add(inputs_3,b3))
        inputs_1 = tf.nn.tanh(tf.nn.bias_add(inputs_1,b1))

        inputs_8 = tf.nn.max_pool(inputs_8,[1,l-7,1,1],[1,1,1,1],padding="VALID")
        inputs_5 = tf.nn.max_pool(inputs_5,[1,l-4,1,1],[1,1,1,1],padding="VALID")
        inputs_3 = tf.nn.max_pool(inputs_3,[1,l-2,1,1],[1,1,1,1],padding="VALID")
        inputs_1 = tf.nn.max_pool(inputs_1,[1,l,1,1],[1,1,1,1],padding="VALID")

        outputs = tf.concat([inputs_8,inputs_5,inputs_3,inputs_1],axis=1)
        return tf.reshape(outputs,[self.batch_size,-1])

      with tf.variable_scope('query_net'):
        source_outputs,ss1,ss2 = tf.contrib.rnn.static_bidirectional_rnn(tf.contrib.rnn.BasicLSTMCell(size),tf.contrib.rnn.BasicLSTMCell(size),emb_inps,dtype=tf.float32)
        #m1 = [tf.reshape(ele,[-1,1,2*size]) for ele in source_outputs]
        #m1 = tf.concat(m1,axis=1)
        #m1 = tf.expand_dims(m1,-1)
        #vec1 = cnn(m1,input_len)
        vec1 = tf.concat([ss1[1],ss2[1]],axis=1)
        print(vec1.get_shape())
      with tf.variable_scope('answer_net'):
        target_outputs,ts1,ts2 = tf.contrib.rnn.static_bidirectional_rnn(tf.contrib.rnn.BasicLSTMCell(size),tf.contrib.rnn.BasicLSTMCell(size),emb_targs,dtype=tf.float32)
        #m2 = [tf.reshape(ele,[-1,1,2*size]) for ele in target_outputs]
        #m2 = tf.concat(m2,axis=1)
        #m2 = tf.expand_dims(m2,-1)
        #vec2 = cnn(m2,output_len)
        vec2 = tf.concat([ts1[1],ts2[1]],axis=1)
        print(vec2.get_shape())
      with tf.variable_scope('loss_calculate'):
        #encoder attention inputs
        
        #vec1 = source_outputs[-1]
        #vec2 = target_outputs[-1]
        #vec = tf.concat([vec1,vec2],axis=1)
        M = tf.get_variable(name='M',shape=[2*size,2*size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='b',shape = [1])
        #w2 = tf.get_variable(name='w2',shape=[size,2],dtype=tf.float32,initializer=tf.truncated_normal_initializer(stddev=0.1))
        #b2 = tf.get_variable(name='b2',shape = [2])
        preds = tf.reduce_sum(tf.multiply(tf.matmul(vec1,M),vec2),axis=1)
        preds = tf.clip_by_value(tf.expand_dims(tf.nn.sigmoid(preds+b),-1),1e-7,1-1e-7)
        preds = tf.concat([preds,1.-preds],axis=1)
        #print(vec1.get_shape())
        #print(vec2.get_shape())
        #pred = tf.matmul(vec,w1)+b1
        #pred = tf.matmul(tf.nn.tanh(pred),w2)+b2
       
        #scores = scores
        #labels = tf.reshape(self.labels,[self.batch_size,1])        
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,logits=preds))
      self.params = []
    for ele in tf.global_variables():
      if ele.name.startswith(name):
        self.params.append(ele)
    opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss,self.params)
    clipped_gradients,norm = tf.clip_by_global_norm(gradients,max_gradient_norm)
    self.update = opt.apply_gradients(zip(clipped_gradients,self.params),global_step=self.global_step)
    self.saver = tf.train.Saver(self.params)
    self.predict = tf.argmax(preds,axis=1)
    tmp = [ele.name for ele in self.params]
    print('\n'.join(tmp))
      

  def step(self,sess,source_inps,target_inps,labels,training=True):
    if len(source_inps)!=self.input_len:
      raise ValueError('input length(%d) must be equal to the input_len(%d)'%(len(inputs),self.input_len))
    #if len(weights)!=self.output_len:
    #  raise ValueError('weight length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    if len(target_inps)!= self.output_len:
      raise ValueError('target length(%d) must be equal to the output_len(%d)'%(len(weights),self.output_len))
    
    input_feed = {}
    for i in range(self.input_len):
      input_feed[self.inputs[i].name] = source_inps[i]
    for i in range(self.output_len):
      input_feed[self.targets[i].name] = target_inps[i]
    input_feed[self.labels.name] = labels
      #input_feed[self.decoder_inputs[i].name] = decoder_inps[i]
    #input_feed[self.targets[-1].name] = np.zeros([self.batch_size],dtype=np.int32)

    if training:
       output_feed = [self.loss,self.update]
    else:
       output_feed = [self.loss,self.predict]
    outputs = sess.run(output_feed,input_feed)
    return outputs[0],outputs[1]

  def get_batch(self, data,batch_size):
    encoder_inputs, decoder_inputs, labels = [], [], []
    for _ in range(batch_size):

        encoder_input, decoder_input = random.choice(data)
        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (self.input_len - len(encoder_input))
        encoder_inputs.append(encoder_input + encoder_pad)

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = self.output_len - len(decoder_input)
        decoder_inputs.append(decoder_input + [data_utils.PAD_ID] * decoder_pad_size)
        
    batch_encoder_inputs, batch_decoder_inputs  = [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(self.input_len):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))
    for length_idx in range(self.output_len):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(batch_size)], dtype=np.int32))
    #print(batch_encoder_inputs,batch_decoder_inputs,labels)
    return batch_encoder_inputs,batch_decoder_inputs

      
