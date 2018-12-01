#coding=utf-8
from __future__ import division

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
import tensorflow as tf
import numpy as np
from cnn import APCNN
from config import FLAGS
import data_loader


def build_vocab(word_list):
    vocab = dict()
    id2tok = dict()
    f = open(word_list, 'r')
    for line in f.readlines():
        line = line.strip("\n")
        token, id = line.split("\t#\t")
        token = token.strip()
        token = token.decode("utf-8")
        id = id.strip()
        id = int(id)
        if token not in vocab:
            vocab[token] = id

        if id not in id2tok:
            id2tok[id] = token.encode("utf-8")
    print "build vocab done"
    return vocab, id2tok


def read_ans(file_name, seq_len):
    ans = []
    f = open(file_name, 'r')
    for line in f.readlines():
        line = line.strip()
        _, stdq = line.split("\t")
        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))

        if stdq not in ans:
            ans.append(stdq)

    print "read alist done"
    return ans


def tok2id(string, seq_len, vocab):
    ids = []
    string = string.strip().strip("_").strip()
    toks = string.split("_")
    for tok in toks:
        id = vocab.get(tok, 0)   #0是<unk>，1是<pad>
        ids.append(id)
    ids = ids[:seq_len]
    ids = ids + [1] * (seq_len - len(ids))

    return ids


def de_id(ids, id2tok):
    toks = []
    ids = [int(id) for id in ids]
    for id in ids:
        tok = id2tok[id]
        toks.append(tok)
    line = " ".join(toks)
    return line


def read_dev(file_name, seq_len):
    dev_data = dict()
    f = open(file_name, 'r')
    for line in f.readlines():
        line = line.strip()
        userq, stdq = line.split("\t")

        userq = userq.strip()

        stdq = stdq.strip().split()
        stdq = stdq[:seq_len]
        stdq = stdq + [1] * (seq_len - len(stdq))

        if userq not in dev_data:   # dict的key不能是list，因此userq保留string的形式
            dev_data[userq] = [stdq]
        else:
            dev_data[userq].append(stdq)

    return dev_data



def dev(ckpt_path, k=30, mode=tf.estimator.ModeKeys.PREDICT):
    print "read data..."
    ans = read_ans("data/id_data_sort", 16)
    dev_data = read_dev("data/id_dev_2w", 16)
    print "read data done"

    with tf.Graph().as_default():
        with tf.device("/gpu:0"):
            session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement)
            session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)

            with sess.as_default():

                model = APCNN(FLAGS, mode)
                saver = tf.train.Saver(tf.global_variables())
                sess.run(tf.global_variables_initializer())
                saver.restore(sess=sess, save_path=ckpt_path)


                cnt = 0
                dev_count = 0
                for userq in dev_data:
                    print "\tEvaluation step:", dev_count
                    dev_count += 1

                    q = userq.strip().split()
                    q = q[:FLAGS.max_sequence_length]
                    q = q + [1] * (FLAGS.max_sequence_length - len(q))
                    devs = []
                    scores = []
                    for a in ans:
                        devs.append((q, a))
                    batches = data_loader.batch_iter(devs, FLAGS.batch_size, 1, False)
                    for batch in batches:
                        feed_dict = {
                            model.usrq: batch[:, 0],
                            model.pos: batch[:, 1],
                            model.dropout_keep_prob: 1.0,
                            model.is_training: False
                        }

                        score = sess.run(model.score, feed_dict)
                        score = tf.reshape(score, [-1])
                        scores.append(score)

                    scores = tf.reshape(scores, [-1])
                    topk = tf.nn.top_k(scores, k)

                    index = sess.run(topk)[1]

                    recalls = np.array(ans)[index]  # 召回的相似Q
                    for recall in recalls:
                        recall = list(recall)
                        if recall in dev_data[userq]:
                            cnt += 1
                            break  # 有一个相似命中了就退出

                return cnt / len(dev_data)



if __name__ == "__main__":
    args = sys.argv
    ckpt = args[1]
    result = dev(ckpt)
    print "Evaluation:", result



