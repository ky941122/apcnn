#coding=utf-8
from __future__ import division

import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            #start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            start_index = end_index-batch_size
            yield shuffled_data[start_index:end_index]



def read_data(filename, seq_len):
    data = []
    f = open(filename, "r")
    for line in f.readlines():
        line = line.strip()
        q, a1, a2 = line.split("\t")

        q = q.strip()
        q = q.split()
        q = q[:seq_len]
        q = q + [1] * (seq_len - len(q))

        a1 = a1.strip()
        a1 = a1.split()
        a1 = a1[:seq_len]
        a1 = a1 + [1] * (seq_len - len(a1))

        a2 = a2.strip()
        a2 = a2.split()
        a2 = a2[:seq_len]
        a2 = a2 + [1] * (seq_len - len(a2))

        data.append((q, a1, a2))

    print "read data done..."
    return data


