#coding=utf-8
from __future__ import division

import os
import time

import tensorflow as tf

from config import FLAGS
from cnn import APCNN
import data_loader
import datetime



def print_args(flags):
    """Print arguments."""
    print("\nParameters:")
    for attr in flags:
        value = flags[attr].value
        print("{}={}".format(attr, value))
    print("")


def train(mode):
    print "Loading data..."
    data = data_loader.read_data(FLAGS.train_file, FLAGS.max_sequence_length)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = APCNN(FLAGS, mode)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.model_name, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            restore = FLAGS.restore_model
            if restore:
                saver.restore(sess, FLAGS.model_path)
                print("*" * 20 + "\nReading model parameters from %s \n" % FLAGS.model_path + "*" * 20)
            else:
                print("*" * 20 + "\nCreated model with fresh parameters.\n" + "*" * 20)


            def train_step(q_batch, pos_batch, neg_batch, epoch):

                """
                A single training step
                """

                feed_dict = {
                    cnn.usrq: q_batch,
                    cnn.pos: pos_batch,
                    cnn.neg: neg_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.is_training: True
                }

                _, step, loss = sess.run([cnn.update, cnn.global_step, cnn.loss], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print "{}: Epoch {} step {}, loss {:g}".format(time_str, epoch, step, loss)

            # Generate batches
            batches = data_loader.batch_iter(data, FLAGS.batch_size, FLAGS.max_epoch, True)

            num_batches_per_epoch = int((len(data)) / FLAGS.batch_size) + 1

            # Training loop. For each batch...
            epoch = 0
            for batch in batches:
                q_batch = batch[:, 0]
                pos_batch = batch[:, 1]
                neg_batch = batch[:, 2]
                train_step(q_batch, pos_batch, neg_batch, epoch)
                current_step = tf.train.global_step(sess, cnn.global_step)

                if current_step % num_batches_per_epoch == 0:
                    epoch += 1

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))



if __name__ == "__main__":
    print_args(FLAGS)

    # Model Preparation
    mode = tf.estimator.ModeKeys.TRAIN

    train(mode)

