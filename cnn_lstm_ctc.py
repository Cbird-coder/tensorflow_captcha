import tensorflow as tf
import hparam_set as h_sets
import utils
import model_helper as _mh

FLAGS = h_sets.FLAGS
num_classes = utils.num_classes

class CNN_LSTM_CTC_CAPTCHA(object):
    def __init__(self, mode):
        self.mode = mode
        # image [batch_size,image_width,image_height,image_channel]
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
        # SparseTensor required by ctc_loss op
        self.labels = tf.sparse_placeholder(tf.int32)
        self._extra_train_ops = []

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summay = tf.summary.merge_all()

    def _build_model(self):
        filters = [1, 64, 128, 128, FLAGS.out_channels]
        strides = [1, 2]

        feature_h = 0
        feature_w = 0

        # CNN part
        with tf.variable_scope('cnn'):
            x = self.inputs
            tf.summary.image('input_image',x)
            print ("cnn input shape: {}".format(x.get_shape().as_list()))
            for i in range(FLAGS.cnn_layer):
                with tf.variable_scope('unit-%d' % (i + 1)):
                    x = _mh.conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
                    x = _mh.batch_norm(self.mode,'bn%d' % (i + 1), x)
                    x = _mh.leaky_relu(x, FLAGS.leakiness)
                    x = _mh.max_pool(x, 2, strides[1])
                    print('cnn layer: {},shape: {}'.format(i,x.get_shape().as_list()))
            _, feature_h, feature_w, _ = x.get_shape().as_list()
        # LSTM part
        with tf.variable_scope('lstm'):
            print ("cnn output shape: {}".format(x.get_shape().as_list()))
            x = tf.transpose(x, [0, 2, 1, 3])  # [batch_size, feature_w, feature_h, FLAGS.out_channels]
            print ("after transpose shape: {}".format(x.get_shape().as_list()))
            # treat `feature_w` as max_timestep in lstm.
            x = tf.reshape(x, [-1, feature_w, feature_h * FLAGS.out_channels])
            print('rnn_layer input shape: {}'.format(x.get_shape().as_list()))
            #compute input data length,shape:[batch_size,data_len]
            len_temp = tf.sign(tf.add(tf.abs(tf.sign(x)),1))
            self.seq_len = tf.cast(tf.reduce_sum(tf.sign(tf.reduce_sum(len_temp,2)),1),tf.int32)
            #build rnn cells,output shape:[batch_size, max_stepsize, num_hidden]
            outputs = _mh.bi_rnns(self.mode,x,self.seq_len)
            print('rnn_layer output shape: {}'.format(outputs.get_shape().as_list()))
            #get hidden size,typical_rnns:FLAGS.num_hidden,bi_rnns:2*FLAGS.num_hidden
            hidden_layers_sizes = int(outputs.shape[-1])
            print ('rnn_layer output hidden size: {}'.format(hidden_layers_sizes))
            # Reshaping to [batch_size * max_stepsize, num_hidden]
            outputs = tf.reshape(outputs, [-1, hidden_layers_sizes])
            #full connect layer
            W = tf.get_variable(name='W_out',
                                shape=[hidden_layers_sizes, num_classes],
                                dtype=tf.float32,
                                initializer=tf.glorot_uniform_initializer())  # tf.glorot_normal_initializer
            b = tf.get_variable(name='b_out',
                                shape=[num_classes],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer())

            self.logits = tf.matmul(outputs, W) + b
            # Reshaping to [batch_size,time_step,num_classes]
            shape = tf.shape(x)
            self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
            print('fc_layer output shape: {}'.format(self.logits.get_shape().as_list())) 
            # Time major
            self.logits = tf.transpose(self.logits, (1, 0, 2))

    def _build_train_op(self):
        if self.mode == 'train':
            self.global_step = tf.train.get_or_create_global_step()
            #ctc loss compute here
            self.loss = tf.nn.ctc_loss(labels=self.labels,
                                   inputs=self.logits,
                                   sequence_length=self.seq_len)

            self.cost = tf.reduce_mean(self.loss)

            tf.summary.scalar('cost', self.cost)

            #update learning rate
            self.lrn_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                                   self.global_step,
                                                   FLAGS.decay_steps,
                                                   FLAGS.decay_rate,
                                                   staircase=True)

            tf.summary.scalar('learning_rate', self.lrn_rate)
            #adam optimizer
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lrn_rate,
                                                beta1=FLAGS.beta1,
                                                beta2=FLAGS.beta2).minimize(self.loss,global_step=self.global_step)
            train_ops = [self.optimizer] + self._extra_train_ops
            self.train_op = tf.group(*train_ops)
        else:
            #ctc beam search here
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits,
                                          self.seq_len,
                                          merge_repeated=False)
            # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len,merge_repeated=False)
            self.dense_decoded = tf.sparse_tensor_to_dense(self.decoded[0], default_value=-1)
