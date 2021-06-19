#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime
import tensorflow_datasets as tfds

from models.ResNet import resnet50
from modules.TFR_load import TFR_load


def main():
    # TODO
    # config.py

    # Hyper params

    BATCH_SIZE     = 256
    NUM_OF_CLASS   = 10
    LEARNING_RATE  = 1e-1
    SCHEDULER      = True
    NUM_TRAIN_DATA = 60000
    NUM_TEST_DATA  = 10000
    EPOCH = 10000

    TR_STEPS_PER_EPOCH = NUM_TRAIN_DATA//BATCH_SIZE
    TE_STEPS_PER_EPOCH = NUM_TEST_DATA//BATCH_SIZE


    # Data loard

    ds_name = 'cifar10'
    builder = tfds.builder(ds_name)

    tr_ds, te_ds = builder.as_dataset(
        split = ['train', 'test'],
        batch_size = BATCH_SIZE,
        shuffle_files = True)
    
    # Set model
    model = resnet50
    
    
    # LR schedule
    if SCHEDULER:
        LEARNING_RATE = tf.keras.optimizers.schedules.CosineDecay(
            LEARNING_RATE,
            decay_steps = TR_STEPS_PER_EPOCH * 50,
            alpha = 1e-4
        )
        # LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
        #     LEARNING_RATE,
        #     decay_steps=TR_STEPS_PER_EPOCH * 10,
        #     decay_rate=0.1
        # )
    
    # Optimizer
    # optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, epsilon=1e-08)
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
    
    # Tensorboard Writer
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    
    # Training loop
    for EP in range(EPOCH):
        EP += 1
        
        loss_value, acc = 0, 0
        for tr_example in tr_ds:
            images, labels = tr_example["image"], tr_example["label"]
            with tf.GradientTape() as tape:
                logits = model(images / 255)
                loss_value += loss(tf.one_hot(labels, 10), logits)
                for layer in model.layers:
                    loss_value += tf.math.reduce_sum(layer.losses)
                
                tr_accuracy.update_state(tf.one_hot(labels, 10), logits)
                acc += tr_accuracy.result()
            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value / TR_STEPS_PER_EPOCH, step=EP)
            tf.summary.scalar('acc', acc / TR_STEPS_PER_EPOCH , step=EP)
            tr_accuracy.reset_state()

            for w in model.weights:
                tf.summary.histogram(w.name, w, step=EP)

        loss_value, acc = 0, 0
        for te_example in te_ds:
            images, labels = te_example["image"], te_example["label"]
            logits = model(images / 255)
            loss_value += loss(tf.one_hot(labels, 10), logits)
            te_accuracy.update_state(tf.one_hot(labels, 10), logits)
            acc += te_accuracy.result()
            
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', loss_value / TE_STEPS_PER_EPOCH, step=EP)
            tf.summary.scalar('acc', acc / TE_STEPS_PER_EPOCH, step=EP)
            te_accuracy.reset_state()


if __name__ == '__main__':
    main()