import datetime
import logging
import os
import sys
import time
import math

import cv2
import numpy as np
import tensorflow as tf

from cnn_lstm_ctc import CNN_LSTM_CTC_CAPTCHA
import utils
import hparam_set as h_set

FLAGS = h_set.FLAGS

logger = logging.getLogger('Traing for CAPTCHA using CNN+LSTM+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train',config=None):
    model = CNN_LSTM_CTC_CAPTCHA(mode)
    model.build_graph()
    train_feeder = utils.DataIterator(data_dir=train_dir)
    print('train size: ', train_feeder.size)
    val_feeder = utils.DataIterator(data_dir=val_dir)
    print('validation size: {}\n'.format(val_feeder.size))

    num_train_samples = train_feeder.size  # 100000
    num_batches_per_epoch = int(math.ceil(float(num_train_samples) / FLAGS.batch_size))  # example: 100000/100

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(math.ceil(float(num_val_samples) / FLAGS.batch_size))  # example: 10000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        saver = utils.resore_or_load_mode(sess,FLAGS)

        print('++++++++++++start training+++++++++++++')
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()

            # the training part
            for cur_batch in range(num_batches_per_epoch):
                batch_time = time.time()
                if (cur_batch + 1) * FLAGS.batch_size <= num_train_samples:
                    indexs = [shuffle_idx[i % num_train_samples] for i in range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                else:
                    indexs = [shuffle_idx[i % num_train_samples] for i in range(cur_batch * FLAGS.batch_size, num_train_samples)]
                batch_inputs, _, batch_labels = train_feeder.input_index_generate_batch(indexs)
                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                # if summary is needed
                summary_str, batch_cost, step, _ = sess.run([model.merged_summay,
                             model.cost,
                             model.global_step,
                             model.train_op], feed_dict=feed)
                # calculate the cost
                train_cost += batch_cost * FLAGS.batch_size
                train_writer.add_summary(summary_str, step)
                
                print('current batch loss:%f'%(batch_cost))

                # save the checkpoint
                if step % FLAGS.save_steps == 0:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save checkpoint at step {0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'captcha'), global_step=step)
                # do validation
                if step % FLAGS.validation_steps == 999:
                    acc_batch_total = 0
                    lastbatch_err = 0
                    lr = 0
                    print('++++++++++++start validation+++++++++++++')
                    for j in range(num_batches_per_epoch_val):
                        if (j + 1) * FLAGS.batch_size <= num_val_samples:
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                        else:
                            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in range(j * FLAGS.batch_size, num_val_samples)]
                        val_inputs, _, val_labels = val_feeder.input_index_generate_batch(indexs_val)
                        
                        val_feed = {model.inputs: val_inputs,
                                    model.labels: val_labels}
                        lastbatch_err, lr = sess.run(
                            [model.cost, model.lrn_rate],
                            feed_dict=val_feed)
                        # print the decode result
                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)
                    # train_err /= num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "avg_train_cost = {:.3f}, " \
                          "lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, avg_train_cost,
                                     lastbatch_err, time.time() - start_time, lr))


def infer(img_path, mode='infer',config=None):
    imgList,names = utils.get_file_name(img_path)
    model = CNN_LSTM_CTC_CAPTCHA(mode)
    model.build_graph()
    batch_size = FLAGS.batch_size
    total_steps = int(math.ceil(float(len(imgList)) / (batch_size)))
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        utils.resore_or_load_mode(sess,FLAGS)
        decoded_expression = []
        print('++++++++++++start inference+++++++++++++')
        for curr_step in range(total_steps):
            imgs_input = []
            for img in imgList[curr_step * batch_size: (curr_step + 1) * batch_size]:
                im = cv2.imread(img, 0).astype(np.float32) / 255.
                im = np.reshape(im, [FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel])
                imgs_input.append(im)

            imgs_input = np.asarray(imgs_input)

            feed = {model.inputs: imgs_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed_dict=feed)

            for item in dense_decoded_code:
                expression = ''
                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)

        with open('./result.txt', 'w') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):
    gpu_conf = tf.GPUOptions(visible_device_list=FLAGS.gpu_id, allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_conf)
    
    if sys.argv[1] == 'train':
        train(FLAGS.train_dir, FLAGS.val_dir, 'train',config)
    elif sys.argv[1] == 'infer':
        infer(FLAGS.infer_dir, 'infer',config)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
