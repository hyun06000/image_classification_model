#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import tensorflow_datasets as tfds

from models.ResBlock import resnet
from modules.TFR_load import TFR_load


def main(_):
    # TODO
    # config.py

    # Hyper params

    BATCH_SIZE     = 128
    NUM_OF_CLASS   = 10
    LEARNING_RATE  = 1e-5
    COSINE_DECAY   = True
    NUM_TRAIN_DATA = 60000
    NUM_TEST_DATA  = 10000
    EPOCH = 100

    TR_STEPS_PER_EPOCH = NUM_TRAIN_DATA//BATCH_SIZE
    TE_STEPS_PER_EPOCH = NUM_TEST_DATA//BATCH_SIZE


    # data loard

    ds_name = 'cifar10'
    builder = tfds.builder(ds_name)

    tr_ds, te_ds = builder.as_dataset(split = ['train', 'test'], shuffle_files = True)

    tr_ds = tr_ds.batch(BATCH_SIZE)
    te_ds = te_ds.batch(BATCH_SIZE)

    model = resnet

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    step = 1
    for EP in range(EPOCH):
        for tr_example, te_example in zip(tr_ds, te_ds):
            images, labels = tr_example["image"], tr_example["label"]
            with tf.GradientTape() as tape:
                logits = model(images / 255)
                loss_value = loss(tf.one_hot(labels, 10), logits)
                accuracy.update_state(tf.one_hot(labels, 10), logits)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_value, step=step)
                    tf.summary.scalar('acc', accuracy.result(), step=step)
                    accuracy.reset_state()
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))


            images, labels = te_example["image"], te_example["label"]
            logits = model(images / 255)
            loss_value = loss(tf.one_hot(labels, 10), logits)
            accuracy.update_state(tf.one_hot(labels, 10), logits)
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', loss_value, step=step)
                tf.summary.scalar('acc', accuracy.result(), step=step)
                accuracy.reset_state()
            step += 1

if __name__ == '__main__':
    main()