# -*- coding: utf-8 -*-
#/usr/bin/python3

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import logging


def load_doc_vocab():
    logging.info("Loading doc vocab...")
    word2idx = {}
    idx2word = {}
    for line in codecs.open(hp.doc_dict, 'r', 'utf-8').readlines():
        idx, word = line.split()
        word2idx[word] = int(idx)
        idx2word[int(idx)] = word
    
    logging.info("Size of doc dict: {}".format(len(word2idx)))
    return word2idx, idx2word


def load_sum_vocab():
    logging.info("Loading sum vocab...")
    word2idx = {}
    idx2word = {}
    for line in codecs.open(hp.sum_dict, 'r', 'utf-8').readlines():
        idx, word = line.split()
        word2idx[word] = int(idx)
        idx2word[int(idx)] = word
          
    logging.info("Size of doc dict: {}".format(len(word2idx)))
    return word2idx, idx2word


def create_data(source_path, target_path):
    logging.info("Creating data...")
    
    article2idx, idx2article = load_doc_vocab()
    sum2idx, idx2sum = load_sum_vocab()
    
    source_file = open(source_path, 'r', encoding='utf-8')
    target_file = open(target_path, 'r', encoding='utf-8')

    X, Y, Sources, Targets = [], [], [], []
    cur_ariticle_idx = 0
    
    while True:
        source_sent = source_file.readline()
        target_sent = target_file.readline()

        if not source_sent:
            if target_sent:
                raise ValueError("inconsistent number of articles in source and target file")
            break

        if cur_ariticle_idx % 1000000 == 0:
            print("\tPreparing {}-th article matrix".format(cur_ariticle_idx))
        
        # if cur_ariticle_idx == 400:
        #     break  # TEMP
        
        source_sent = source_sent.split()
        target_sent = target_sent.split()

        # remove short sentences & chop long sentences
        if len(source_sent) < hp.article_minlen or len(target_sent) < hp.summary_minlen:
            continue

        if len(source_sent) >= hp.article_maxlen:
            source_sent = source_sent[:(hp.article_maxlen-1)] # 1 for </S>

        if len(target_sent) >= hp.summary_maxlen:
            target_sent = target_sent[:(hp.summary_maxlen-1)]

        x = [article2idx.get(word, 1) for word in (source_sent + [u"</S>"])]
        y = [sum2idx.get(word, 1) for word in (target_sent + [u"</S>"]) ]
        
        if len(x) < hp.article_maxlen:
            x = np.lib.pad(x, [0, hp.article_maxlen - len(x)], 'constant', constant_values=(0, 0))
        if len(y) < hp.summary_maxlen:
            y = np.lib.pad(y, [0, hp.summary_maxlen - len(y)], 'constant', constant_values=(0, 0))

        try:
            assert len(x) == hp.article_maxlen
            assert len(y) == hp.summary_maxlen
        except AssertionError as error:
            print("current article length: ", len(x), "current article maxlen: ", hp.article_maxlen)
            print("current summary length: ", len(y), "current summary maxlen: ", hp.summary_maxlen)

        X.append(x)
        Y.append(y)
        Sources.append(" ".join(source_sent).strip())
        Targets.append(" ".join(target_sent).strip())
        
        cur_ariticle_idx += 1
    
    source_file.close()
    target_file.close()

    X = np.array(X)
    Y = np.array(Y)
    print("number of data: ", X.shape, Y.shape)
    return X, Y, Sources, Targets


def create_test_data(source_sents):
    print("Creating data...")
    article2idx, idx2article = load_doc_vocab()

    doc_sents = list(map(lambda line: line.split(), source_sents))
    
    # Index
    X, Sources = [], []

    cur_ariticle_idx = 0
    for source_sent in doc_sents:
        if cur_ariticle_idx % 100000 == 0:
            print("\tPreparing {}-th article matrix".format(cur_ariticle_idx))

        # if cur_ariticle_idx == 200:
        #   break  # TEMP

        x = [article2idx.get(word, 1) for word in (source_sent + [u"</S>"]) ]

        if len(x) <= hp.article_maxlen:
            x = np.lib.pad(x, [0, hp.article_maxlen - len(x)], 'constant', constant_values=(0, 0))

            try:
                assert len(x) == hp.article_maxlen
            except AssertionError as error:
                print("current article length: ", len(x), "current article maxlen: ", hp.article_maxlen)

            X.append(x)
            Sources.append(" ".join(source_sent).strip())
        cur_ariticle_idx += 1
    X = np.array(X)
    return X, Sources

def load_data(type='train'):
    LEGAL_TYPE = ('train', 'eval', 'test', 'eval_tmp')
    if type not in LEGAL_TYPE:
        raise TypeError('Invalid type: should be train/test/eval')
    
    if type == 'train' or type == 'eval_tmp':
        doc_path = hp.source_train
        sum_path = hp.target_train
    elif type == 'eval':
        doc_path = hp.source_valid
        sum_path = hp.target_valid
    elif type == 'test':
        doc_path = hp.source_test
    
    
    if type == 'test':
        with open(doc_path, 'r', encoding="utf-8") as docfile:
            doc_sents = docfile.readlines()
        X, Sources = create_test_data(doc_sents)
        return X, Sources # (1064, 150)

    else:
        X, Y, Sources, Targets = create_data(doc_path, sum_path)
        if type == 'train':
            return X, Y
        elif type == 'eval':
            return X, Sources, Targets
        elif type == 'eval_tmp':
            return X, Sources, Targets


def get_batch_data():
    print("getting batch_data...")
    X, Y = load_data(type='train')
    
    num_batch = len(X) // hp.batch_size
    
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    input_queues = tf.train.slice_input_producer([X, Y])    # Produces a slice of each `Tensor` in `tensor_list`
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)

    return x, y, num_batch # (N, T), (N, T), ()


if __name__ == '__main__':
    X, Sources, Targets = load_data(type='eval')
    print("Sources: ", len(Sources), "Targets: ", len(Targets))
    
    for source, target in zip(Sources, Targets):
        print("source: ", source)
        print("target: ", target)
        print()
