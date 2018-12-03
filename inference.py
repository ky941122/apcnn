#coding=utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import jieba
import tensorflow as tf
import numpy as np

import data_loader
from cnn import APCNN
from config import FLAGS


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


def inference(word_list, user_dict, train, ckpt_path, k=30, mode=tf.estimator.ModeKeys.PREDICT):
    k = int(k)
    tokenizer = jieba.Tokenizer()
    tokenizer.load_userdict(user_dict)
    vocab, id2tok = build_vocab(word_list)   #vocab里token是unicode， id2tok里tok是str， 两个里面id都是int
    print "read data"
    alist = read_ans(train, FLAGS.max_sequence_length)  #是个二维list

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

                while True:
                    print "Please input query:"
                    line = sys.stdin.readline().strip()
                    if not line:
                        line = "小米蓝牙手柄能连接手机玩吗"
                    ws = tokenizer.cut(line)  #切出来每个tok是unicode。
                    ws = list(ws)
                    q = "_".join(ws)

                    ws_enc = [tok.encode("utf-8") for tok in ws]
                    q_enc = "_".join(ws_enc)

                    print "tokenized query is:", q_enc

                    q = tok2id(q, FLAGS.max_sequence_length, vocab)  #是个list
                    print "id q is:", q

                    devs = []
                    scores = []
                    for a in alist:
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

                    recalls = np.array(alist)[index]  # 召回的相似Q

                    print "Recall results are: \n"
                    for recall in recalls:
                        line = de_id(recall, id2tok)
                        print line, "\n"
                    


if __name__ == "__main__":
    word_list = "data/word_list"
    user_dict = "data/userterms.dic"
    train = 'data/id_data_sort'
    args = sys.argv
    ckpt_path = args[1]
    if len(args) > 2:
        k = args[2]
        if len(args) > 3:
            mode = args[3]
            inference(word_list, user_dict, train, ckpt_path, k, mode)
        else:
            inference(word_list, user_dict, train, ckpt_path, k)
    else:
        inference(word_list, user_dict, train, ckpt_path)

