#!/usr/bin/env python
# coding: utf-8
# @Author: lapis-hong
# @Date  : 2018/8/6
"""This module contains model hyper parameters using tf.flags."""
import tensorflow as tf

# data params
tf.flags.DEFINE_string("train_file", 'data/id_data_with_neg_1', "train file")
# tf.flags.DEFINE_string("test_file", '', "test file")
# tf.flags.DEFINE_string("vocab_file", 'data/WikiQA/vocab', "vocab file")
# tf.flags.DEFINE_string("embed_file", None, "embed file")
# tf.flags.DEFINE_string("predict_file", '', "predict file")
# tf.flags.DEFINE_string("output_file", 'result.txt', "output file")

tf.flags.DEFINE_integer("max_sequence_length", 16, "max question length [40]")
# tf.flags.DEFINE_integer("question_max_len", 40, "max question length [40]")
# tf.flags.DEFINE_integer("answer_max_len", 80, "max answer length [80]")
# tf.flags.DEFINE_integer("num_buckets", 1, "buckets of sequence length [1]")
tf.flags.DEFINE_integer("embedding_dim", 300, "embedding dim [100/300]")
# tf.flags.DEFINE_integer("shuffle_buffer_size", 10000, "Shuffle buffer size")

# model params
# tf.flags.DEFINE_integer("model_type", 1, "model type, 1 for AP-CNN, 2 for AP-biLSTM [1]")
# tf.flags.DEFINE_string("model_dir", "model", "model path")
# common
tf.flags.DEFINE_float("margin", 2.5, "loss function margin, 0.5 for AP-CNN, 0.2 for AP-BiLSTM")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "dropout keep prob [0.8]")
# cnn
tf.flags.DEFINE_integer("num_filters", 200, "num of conv filters [400]")
tf.flags.DEFINE_string("filter_sizes_1", '1,2,3,4', "filter sizes")
tf.flags.DEFINE_string("filter_sizes_2", '3,5,7', "filter sizes")
# lstm
# tf.flags.DEFINE_integer("num_layers", 2, "num of hidden layers [2]")
# tf.flags.DEFINE_integer("hidden_units", 141, "num of hidden units [141]")

# training params
tf.flags.DEFINE_integer("batch_size", 512, "train batch size [20]")
tf.flags.DEFINE_integer("max_epoch", 10000, "max epoch [10]")
tf.flags.DEFINE_float("lr", 0.002, "init learning rate [adam: 0.002, sgd: 1.1]")
tf.flags.DEFINE_integer("lr_decay_epoch", 3, "learning rate decay interval [3]")
tf.flags.DEFINE_float("lr_decay_rate", 0.5, "learning rate decay rate [0.5]")
tf.flags.DEFINE_string("optimizer", "adam", "optimizer, `adam` | `rmsprop` | `sgd` [adam]")
tf.flags.DEFINE_integer("stats_per_steps", 10, "show train info steps [100]")
tf.flags.DEFINE_integer("save_per_epochs", 1, "every epochs to save model [1]")
tf.flags.DEFINE_boolean("use_learning_decay", True, "use learning decay or not [True]")
tf.flags.DEFINE_boolean("use_grad_clip", True, "whether to clip grads [False]")
tf.flags.DEFINE_integer("grad_clip_norm", 5, "max grad norm if use grad clip [5]")
# tf.flags.DEFINE_integer("num_keep_ckpts", 5, "max num ckpts [5]")
# tf.flags.DEFINE_integer("random_seed", 123, "random seed [123]")
tf.flags.DEFINE_integer("num_checkpoints", 2000, "Number of checkpoints to store (default: 2)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")

# auto params, do not need to set
tf.flags.DEFINE_integer("vocab_size", 22511, "vocabulary size")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Model Saving Parameters
tf.flags.DEFINE_boolean("restore_model", False, "Whether restore model or create new parameters")
tf.flags.DEFINE_string("model_path", "runs", "Restore which model")
tf.flags.DEFINE_string("model_name", "model", "Store model in which name")


FLAGS = tf.flags.FLAGS
