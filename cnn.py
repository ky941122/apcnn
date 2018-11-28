#coding=utf-8
from __future__ import division

import tensorflow as tf
import numpy as np


class APCNN:
    def __init__(self, params, mode):
        """Initialize model, build graph.
        Args:
          iterator: instance of class BatchedInput, defined in dataset.
          params: parameters.
          mode: train | eval | predict mode defined with tf.estimator.ModeKeys.
        """
        self.params = params
        self.mode = mode
        self.scope = self.__class__.__name__  # instance class name

        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.usrq = tf.placeholder(tf.int32, [None, None], name="usrq")
            self.pos = tf.placeholder(tf.int32, [None, None], name="pos")
            self.neg = tf.placeholder(tf.int32, [None, None], name="neg")
            self.dropout_keep_prob = tf.placeholder_with_default(1.0, [], name="dropout_keep_prob")
            self.is_training = tf.placeholder_with_default(False, [], name="is_training")

            self.initializer = tf.keras.initializers.he_normal()

            with tf.device('/cpu:0'), tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
                self.embeddings = tf.get_variable("embed_W_words", [params.vocab_size, params.embedding_dim])


            self.q = tf.nn.embedding_lookup(self.embeddings, self.usrq)  # [batch_size, seq_length, embedding_size]
            self.a1 = tf.nn.embedding_lookup(self.embeddings, self.pos)
            # Q, A1, A2 are CNN or biLSTM encoder outputs for question, positive answer, negative answer.
            self.Q = self._encode(self.q)  # b * m * c
            self.A1 = self._encode(self.a1)  # b * n * c
            self.r_q, self.r_a1 = self._attentive_pooling(self.Q, self.A1)
            self.score = self._cosine(self.r_q, self.r_a1)
            self._model_stats()  # print model statistics info

            if mode != tf.estimator.ModeKeys.PREDICT:
                self.a2 = tf.nn.embedding_lookup(self.embeddings, self.neg)
                self.A2 = self._encode(self.a2)  # b * n * c
                self.r_q, self.r_a2 = self._attentive_pooling(self.Q, self.A2)
                self.negative_score = self._cosine(self.r_q, self.r_a2)

                with tf.name_scope("loss"):
                    self.loss = tf.reduce_mean(
                        tf.maximum(0.0, self.params.margin - self.score + self.negative_score))

                    if params.optimizer == "rmsprop":
                        opt = tf.train.RMSPropOptimizer(params.lr)
                    elif params.optimizer == "adam":
                        opt = tf.train.AdamOptimizer(params.lr)
                    elif params.optimizer == "sgd":
                        opt = tf.train.MomentumOptimizer(params.lr, 0.9)
                    else:
                        raise ValueError("Unsupported optimizer %s" % params.optimizer)
                    train_vars = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, train_vars)
                    # gradients, _ = opt.compute_gradients(self.loss, train_vars)
                    if params.use_grad_clip:
                        gradients, grad_norm = tf.clip_by_global_norm(
                            gradients, params.grad_clip_norm)

                    self.global_step = tf.Variable(0, trainable=False)
                    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    with tf.control_dependencies(update_op):
                        self.update = opt.apply_gradients(
                            zip(gradients, train_vars), global_step=self.global_step)


    def _encode(self, x):
        params = self.params

        # Use tf high level API tf.layers

        conv1_outputs = []
        for i, filter_size in enumerate(map(int, params.filter_sizes_1.split(','))):
            with tf.variable_scope("1st_conv_{}".format(filter_size)):
                filter_shape = [filter_size, params.embedding_dim, params.num_filters]
                W = tf.get_variable("first_conv_{}_W".format(filter_size), shape=filter_shape, initializer=self.initializer)

                conv1 = tf.nn.conv1d(x, W, stride=1, padding="SAME", name="first_conv")  # (batch_size, seq_length，num_filters)
                print "conv1 shape:", conv1.shape
                conv1 = tf.layers.batch_normalization(conv1, axis=-1, training=self.is_training)  # axis定的是channel在的维度。
                h = tf.nn.relu(conv1, name="relu_1")
                conv1_outputs.append(h)

        conv2_inputs = tf.concat(conv1_outputs, -1)
        print "conv2 inputs shape:", conv2_inputs.shape


        conv2_outputs = []
        for i, filter_size in enumerate(map(int, params.filter_sizes_2.split(','))):
            with tf.variable_scope("second_conv_maxpool_%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, params.num_filters * len(params.filter_sizes_1.split(',')), params.num_filters]
                W = tf.get_variable("second_conv_{}_W".format(filter_size), shape=filter_shape, initializer=self.initializer)
                conv2 = tf.nn.conv1d(
                    conv2_inputs,
                    W,
                    stride=1,
                    padding="SAME",
                    name="second_conv")
                conv2 = tf.layers.batch_normalization(conv2, axis=-1, training=self.is_training)  # axis定的是channel在的维度。
                h = tf.nn.relu(conv2, name="relu_2")
                conv2_outputs.append(h)

        # Combine all the features
        outputs = tf.concat(conv2_outputs, 2, name="output")  # (batch_size, seq_length, num_filters_total)
        print "outputs shape:", outputs.shape
        if self.mode == tf.estimator.ModeKeys.TRAIN:
            outputs = tf.nn.dropout(outputs, self.dropout_keep_prob, name="output")

        return outputs


    def _attentive_pooling(self, q, a):
        """Attentive pooling
        Args:
            q: encoder output for question (batch_size, q_len, vector_size)
            a: encoder output for question (batch_size, a_len, vector_size)
        Returns:
            final representation Tensor r_q, r_a for q and a (batch_size, vector_size)
        """
        batch_size = self.params.batch_size
        c = q.get_shape().as_list()[-1]  # vector size
        with tf.variable_scope("attentive-pooling") as scope:
            # G = tanh(Q*U*A^T)  here Q is equal to Q transpose in origin paper.
            self.Q = q  # (b, m, c)
            self.A = a  # (b, n, c)
            self.U = tf.get_variable(
                "U", [c, c],
                initializer=tf.truncated_normal_initializer(stddev=0.1)
            )
            self.U_batch = tf.tile(tf.expand_dims(self.U, 0), [batch_size, 1, 1])
            self.G = tf.tanh(
                tf.matmul(
                    tf.matmul(self.Q, self.U_batch), tf.transpose(self.A, [0, 2, 1]))
            )  # G b*m*n

            # column-wise and row-wise max-poolings to generate g_q (b*m*1), g_a (b*1*n)
            g_q = tf.reduce_max(self.G, axis=2, keepdims=True)
            g_a = tf.reduce_max(self.G, axis=1, keepdims=True)

            # create attention vectors sigma_q (b*m), sigma_a (b*n)
            sigma_q = tf.nn.softmax(g_q)
            sigma_a = tf.nn.softmax(g_a)
            # final output r_q, r_a  (b*c)
            r_q = tf.squeeze(tf.matmul(tf.transpose(self.Q, [0, 2, 1]), sigma_q), axis=2)
            r_a = tf.squeeze(tf.matmul(sigma_a, self.A), axis=1)
            return r_q, r_a  # (b, c)

    @staticmethod
    def _cosine(x, y):
        """x, y shape (batch_size, vector_size)"""
        # normalize_x = tf.nn.l2_normalize(x, 0)
        # normalize_y = tf.nn.l2_normalize(y, 0)
        # cosine = tf.reduce_sum(tf.multiply(normalize_x, normalize_y), 1)
        cosine = tf.div(
            tf.reduce_sum(x * y, 1),
            tf.sqrt(tf.reduce_sum(x * x, 1)) * tf.sqrt(tf.reduce_sum(y * y, 1)) + 1e-8,
            name="cosine")
        return cosine

    @staticmethod
    def _model_stats():
        """Print trainable variables and total model size."""

        def size(v):
            return reduce(lambda x, y: x * y, v.get_shape().as_list())

        print("Trainable variables")
        for v in tf.trainable_variables():
            print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
        print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))



