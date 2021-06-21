#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import tensorflow_datasets as tfds

from models.ResNet import ResNet
from modules.TFR_load import random_flip_and_crop
from modules.trainer import train_loop, data_whitening


def main():
    # TODO
    # config.py

    # Hyper params
    
    SEED               = 1
    
    BATCH_SIZE         = 128
    NUM_OF_CLASS       = 10
    LEARNING_RATE      = 1e-1
    OPTMIZER           = 'SGD'
    SCHEDULER          = 'custom'
    NUM_TRAIN_DATA     = 50000
    NUM_TEST_DATA      = 10000
    EPOCH              = 10000

    TR_STEPS_PER_EPOCH = NUM_TRAIN_DATA//BATCH_SIZE
    TE_STEPS_PER_EPOCH = NUM_TEST_DATA//BATCH_SIZE
    
    GRAPH              = False #True
    HIST_LOG           = False #True
    
    # Random seed
    tf.random.set_seed(SEED)
    
    # Data loard

    ds_name = 'cifar10'
    builder = tfds.builder(ds_name)

    tr_ds, te_ds = builder.as_dataset(
        split = ['train', 'test']
    )
    
    tr_ds = tr_ds\
            .map(random_flip_and_crop)\
            .repeat()\
            .shuffle(NUM_TRAIN_DATA, reshuffle_each_iteration=True)\
            .batch(BATCH_SIZE)\
            .prefetch(tf.data.AUTOTUNE)
    te_ds = te_ds\
            .repeat()\
            .batch(BATCH_SIZE)\
            .prefetch(tf.data.AUTOTUNE)
    
    # Set model
    model = ResNet()
    
    if GRAPH:
        model.trace_graph([100,32,32,3])
    
    # LR schedule
    if SCHEDULER:
        if SCHEDULER == 'cosine':
            LEARNING_RATE = tf.keras.optimizers.schedules.CosineDecay(
                LEARNING_RATE,
                decay_steps = TR_STEPS_PER_EPOCH * 5,
                alpha = 1e-3
            )
        elif SCHEDULER == 'exponentioal':
            LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
                LEARNING_RATE,
                decay_steps=500,
                decay_rate=0.1,
                staircase=True
            )
    
    # Optimizer
    if OPTMIZER:
        if OPTMIZER == 'Adam' or OPTMIZER == 'ADAM' or OPTMIZER == 'adam' :
            optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-07)
        elif OPTMIZER == 'sgd' or OPTMIZER == 'SGD' : 
            optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE,momentum=0.9)
    
    # Loss function
    loss = tf.keras.losses.CategoricalCrossentropy()
    
    
    # Metric
    tr_accuracy = tf.keras.metrics.CategoricalAccuracy()
    te_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    # Log config
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    
    train_loop(
        "steps",# log_freq,
        train_log_dir,
        test_log_dir,
        tr_ds,
        te_ds,
        model,
        loss,
        optimizer,
        tr_accuracy,
        te_accuracy,
        EPOCH,
        TR_STEPS_PER_EPOCH,
        TE_STEPS_PER_EPOCH,
        HIST_LOG,
        SCHEDULER
    )
    
if __name__ == '__main__':
    main()