import tensorflow as tf
import hparam_set as h_sets

FLAGS = h_sets.FLAGS

########rnn relate function######
def create_single_layer_rnn():
	return tf.nn.rnn_cell.LSTMCell(FLAGS.num_hidden, state_is_tuple=True)

def create_rnn_layers(mode):
	cells = []
	for i in range(FLAGS.rnn_layer):
		cell = create_single_layer_rnn()
		if mode=='train':
			cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=FLAGS.output_keep_prob)
		cells.append(cell)

	rnn_cells = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
	return rnn_cells

def typical_rnns(mode,x,seq_len):

	rnn_cells = create_rnn_layers(mode)

	outputs, _ = tf.nn.dynamic_rnn(
                cell=rnn_cells,
                inputs=x,
                sequence_length=seq_len,
                dtype=tf.float32,
                time_major=FLAGS.time_major
            )
	return outputs

def bi_rnns(mode,x,seq_len):
	fw_cell = create_rnn_layers(mode)
	bw_cell = create_rnn_layers(mode)
	rnn_output,_ = tf.nn.bidirectional_dynamic_rnn(fw_cell,
													bw_cell,
													x,
													scope = 'bi_lstm',
													dtype = tf.float32,
													sequence_length = seq_len,
													time_major=FLAGS.time_major)
	rnn_output = tf.concat(rnn_output,axis=2)
	return rnn_output

#cnn layer relate code
def conv2d(x, name, filter_size, in_channels, out_channels, strides):
	with tf.variable_scope(name):
		kernel = tf.get_variable(name='W',
			shape=[filter_size, filter_size, in_channels, out_channels],
			dtype=tf.float32,
			initializer=tf.glorot_uniform_initializer())

		b = tf.get_variable(name='b',
			shape=[out_channels],
			dtype=tf.float32,
			initializer=tf.constant_initializer())

		con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

	return tf.nn.bias_add(con2d_op, b)

def batch_norm(mode,name, x):
	"""Batch normalization."""
	with tf.variable_scope(name):
		x_bn = tf.contrib.layers.batch_norm(
			inputs=x,
			decay=0.9,
			center=True,
			scale=True,
			epsilon=1e-5,
			updates_collections=None,
			is_training=mode == 'train',
			fused=True,
			data_format='NHWC',
			zero_debias_moving_mean=True,
			scope='BatchNorm'
			)
		return x_bn

def leaky_relu(x, leakiness=0.0):
	return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def max_pool(x, ksize, strides):
	return tf.nn.max_pool(x,
		ksize=[1, ksize, ksize, 1],
		strides=[1, strides, strides, 1],
		padding='SAME',
		name='max_pool')