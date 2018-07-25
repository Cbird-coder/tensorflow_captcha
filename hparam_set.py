import tensorflow as tf

tf.app.flags.DEFINE_string('checkpoint_dir', './checkpoint/', 'the checkpoint dir')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001, 'inital lr')

tf.app.flags.DEFINE_integer('image_height', 60, 'image height')
tf.app.flags.DEFINE_integer('image_width', 160, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')

tf.app.flags.DEFINE_integer('cnn_layer', 4, 'cnn layers for extract image feature')
tf.app.flags.DEFINE_integer('out_channels', 64, 'output channels of last layer in CNN')

tf.app.flags.DEFINE_integer('rnn_layer', 2, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_hidden', 256, 'number of hidden units in lstm')
tf.app.flags.DEFINE_boolean('time_major', False, 'number of hidden units in lstm')
tf.app.flags.DEFINE_float('output_keep_prob', 0.8, 'output_keep_prob in lstm')

tf.app.flags.DEFINE_integer('num_epochs', 10000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size', 320, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps', 500, 'the step to save checkpoint')
tf.app.flags.DEFINE_float('leakiness', 0.01, 'leakiness of lrelu')
tf.app.flags.DEFINE_integer('validation_steps', 1000, 'the step to validation')
tf.app.flags.DEFINE_integer('save_max_time', 10, 'the maximum time save models')

tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')

tf.app.flags.DEFINE_string('train_dir', './img/train/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir', './img/val/', 'the val data dir')
tf.app.flags.DEFINE_string('infer_dir', './img/infer/', 'the infer data dir')
tf.app.flags.DEFINE_string('log_dir', './log', 'the logging dir')
tf.app.flags.DEFINE_string('gpu_id', '1', 'the gpu you used')

FLAGS = tf.app.flags.FLAGS