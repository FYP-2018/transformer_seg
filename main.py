# -*- coding: utf-8 -*-
#/usr/bin/python3
# TRANSFORMER + RL (nsml version)

from __future__ import print_function
import os, codecs
import logging

import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu

from modules import *
from graph import Graph
from hyperparams import Hyperparams as hp
from data_load import load_doc_vocab, load_sum_vocab, load_data
from rouge_tensor import rouge_l_sentence_level

def train():
    try:
        if not os.path.exists(hp.logdir):
            os.mkdir(hp.logdir)
    except:
        tf.logging.info('making logdir failed')

    # Load vocabulary
    de2idx, idx2de = load_doc_vocab()
    en2idx, idx2en = load_sum_vocab()

    print("Constructing graph...")
    train_g = Graph("train")

    print("Start training...")
    with train_g.graph.as_default():
        saver = tf.train.Saver(max_to_keep=1)
        # sv = tf.train.Supervisor(logdir=hp.logdir, saver=None)
        sv = tf.train.Supervisor(logdir=hp.logdir, saver=saver)

        config = tf.ConfigProto(allow_soft_placement=True,
                                log_device_placement=False)
        config.gpu_options.allow_growth = True

        with sv.managed_session(config=config) as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            print("Start training: epoches={}, num batch={}".format(hp.num_epochs, train_g.num_batch))
            for epoch in range(1, hp.num_epochs+1):
                print("Starting {}-th epoch".format(epoch))

                if sv.should_stop():
                    break

                for step in range(train_g.num_batch):
                    true_step = step + (epoch - 1) * train_g.num_batch

                    if true_step % hp.train_record_steps == 0:
                        summary, _ = sess.run([train_g.merged, train_g.train_op_ml])
                        train_g.filewriter.add_summary(summary, true_step)
                    else:
                        sess.run(train_g.train_op_ml)

                    if true_step % hp.checkpoint_steps == 0:
                        # sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_step_%d' % (epoch, true_step))
                        saver.save(sess, hp.logdir + '/model_epoch_%02d_step_%d' % (epoch, true_step))

                    if true_step > 0 and true_step % hp.eval_record_steps == 0:
                        eval(cur_step=true_step, write_file=False)

                    # iteration indent
                # epoch indent
                # eval(cur_step=true_step, write_file=True)
    print("Done")


def test(num_epoch=10):
    # Load graph
    g = Graph(is_training=False)
    print("Test Graph loaded")

    # Load data
    X, Sources = load_data(type='test')
    word2idx, idx2word = load_sum_vocab()

    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor(logdir=hp.logdir)
        # with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        with sv.managed_session(
            start_standard_services=False,
            config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        ) as sess:

            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

            ## Inference
            if not os.path.exists('results'):
                os.mkdir('results')

            with codecs.open("results/test-" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):
                    ### Get mini-batches
                    x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                    sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]

                    ### Autoregressive inference
                    # preds = np.zeros((hp.batch_size, hp.summary_maxlen), np.int32)
                    preds = np.zeros((x.shape[0], hp.summary_maxlen), np.int32)
                    for j in range(hp.summary_maxlen):
                        # _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        _preds, _acc = sess.run([g.preds, g.acc], {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    print("at step {}: rough-1 = {}".format(num_epoch, _acc))

                    for source, pred in zip(sources, preds): # sentence-wise
                        got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()
                        sentence_to_write = "- source: " + source + "\n- got: " + got + "\n\n"

                        print(sentence_to_write)
                        fout.write(sentence_to_write)
                        fout.flush()


def eval(type='eval', cur_step=0, write_file=True):
    # Load graph
    g = Graph(is_training=False)
    print("Eval Graph loaded")

    # Load data
    # X, Sources, Targets = load_data(type='eval')
    X, Sources, Targets = load_data(type=type)

    de2idx, idx2de = load_doc_vocab()
    word2idx, idx2word = load_sum_vocab()


    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name

            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            if write_file:
                ### Write to file
                fout2 = codecs.open("results/eval-pred", "w", "utf-8")
                fout3 = codecs.open("results/eval-title", "w", "utf-8")
                fout =  codecs.open("results/eval-" + type + "_" + mname, "w", "utf-8")

            list_of_refs, hypotheses = [], []
            num_batch = len(X) // hp.batch_size
            print("num batch: ", num_batch, "len(X): ", len(X))
            
            for i in range(num_batch):
                print("the {} th set of batch ".format(i))
                ### Get mini-batches
                x = X[i*hp.batch_size: (i+1)*hp.batch_size]
                sources = Sources[i*hp.batch_size: (i+1)*hp.batch_size]
                targets = Targets[i*hp.batch_size: (i+1)*hp.batch_size]

                ### Autoregressive inference
                preds = np.zeros((hp.batch_size, hp.summary_maxlen), np.int32)

                for j in range(hp.summary_maxlen):
                    if j % 10 == 0: 
                        print("generating {} prediction position".format(j))

                    if j == hp.summary_maxlen - 1:  # record the accuracy at the end of inference
                        print("running summary")
                        _preds, summary = sess.run([g.preds, g.merged], {g.x: x, g.y: preds})
                        g.filewriter.add_summary(summary, cur_step)
                    else:
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})

                    preds[:, j] = _preds[:, j]

                for source, target, pred in zip(sources, targets, preds): # sentence-wise
                    got = " ".join(idx2word[idx] for idx in pred).split("</S>")[0].strip()

                    if write_file:
                        sentence_to_write = "-source: {}\n-expected: {}\n-got: {}\n\n".format(source, target, got)

                        print(sentence_to_write)
                        fout.write(sentence_to_write)
                        fout2.write(got.strip() + '\n')
                        fout3.write(target.strip() + '\n')

                        fout.flush()
                        fout2.flush()
                        fout3.flush()

                    # bleu score
                    ref = target.split()
                    hypothesis = got.split()
                    if len(ref) > 3 and len(hypothesis) > 3:
                        list_of_refs.append(ref)
                        hypotheses.append(hypothesis)

            ## Calculate bleu score and rouge
            rouge = rouge_l_sentence_level(hypotheses, list_of_refs)
            # (eval_sentences, ref_sentences):

            rouge = np.mean(rouge)

    return None


if __name__ == '__main__':
    logging.info("START")
    logging.info("tensorflow version: ", tf.__version__)

    from tensorflow.python.client import device_lib as dl
    logging.info("local device: ", dl.list_local_devices())

    train()
    # eval()
