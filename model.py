#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds


from preprocessing.data.imagenet_labels import imagenet_labels

from modules.modified_ViT_Layers import PatchEmbedding, ClassToken, Epos, ViT_Block

def decode_fn(example):
    image = example["image"]
    image = tf.cast(image, tf.float32)
    label = example["label"]
    label = tf.one_hot(label,10)
    return image, label

batch_size = 256
ds_name = 'cifar10'
tr_ds, te_ds = tfds.load(ds_name,
                         split = ['train', 'test'],
                         shuffle_files = True)

tr_ds = tr_ds.map(decode_fn)\
                .repeat()\
                .batch(batch_size)\
                .prefetch(tf.data.AUTOTUNE)
te_ds = te_ds.map(decode_fn)\
                .repeat()\
                .batch(batch_size)\
                .prefetch(tf.data.AUTOTUNE)


class MyModel(tf.keras.Model):
    def __init__(self, batch_size):
        super(MyModel, self).__init__()
        
        self.i = 1
        self.batch_size = batch_size
        
        self.PatchEmbedding = PatchEmbedding(D = 128,
                                             patch_size = 16)
        #self.ClassToken = ClassToken()
        #self.Epos = Epos()
        
        num_of_block = 16
        
        self.Block = []
        for _ in range(num_of_block):
            self.Block.append(ViT_Block(hidden_size = 128))
        
        self.flatten = tf.keras.layers.Flatten()
        
        self.dense = tf.keras.layers.Dense(
                                units = 10, # of class
                                activation=None,
                                use_bias=True,
                                kernel_regularizer = tf.keras.regularizers.l2(0.1))

        self.softmax = tf.keras.layers.Softmax(axis=-1)
        #self.LN = tf.keras.layers.LayerNormalization(axis = -1,epsilon=1e-8)
        
    def call(self, x):
        
        print('EPOCH ::: ',self.i)
        self.i += 1
        #CIFAR-10
        x = tf.ensure_shape(x, [self.batch_size,32,32,3])
        
        x = self.PatchEmbedding(x)
        print('input ::: ', x)
        
        for block in self.Block:
            x = block(x)
        
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x

model = MyModel(batch_size)



#decay_steps = 50000//batch_size * 100
#initial_learning_rate = 1e-3
#lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
#    initial_learning_rate, decay_steps,alpha=0.01)
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate = 1e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/' + current_time

tb_cb = tf.keras.callbacks.TensorBoard(
    log_dir=train_log_dir,
    update_freq='epoch')

model.fit(
    tr_ds,
    batch_size=batch_size,
    epochs=50000,
    callbacks=[tb_cb],
    validation_data = te_ds,
    steps_per_epoch = 50000//batch_size,
    validation_steps = 10000//batch_size,
    verbose = 0
)


# Calling `save('my_model.h5')` creates a h5 file `my_model.h5`.
model.save("my_h5_model.h5")

# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_h5_model.h5")


#batch_size=128,
'''
model.fit_generator(
    generator = train_ds, steps_per_epoch=100, epochs=10, verbose=1, callbacks=[tb_cb],
    validation_data=val_ds, validation_steps=100, validation_freq=1,
    class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False,
    shuffle=True, initial_epoch=0
)
'''


'''
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
#loss = tf.keras.losses.MeanSquaredError()

# In[ ]:

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
train_gph_dir = 'logs/graph/' + current_time + '/'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
#writer = tf.summary.create_file_writer(train_gph_dir)

with tf.compat.v1.Session() as sess:
    summary_writer = tf.compat.v2.train.SummaryWriter(train_gph_dir, sess.graph)

i = 0
for images,label in ds:
    with tf.GradientTape() as tape:
        logits = model(images)
        loss_value = loss(logits, tf.one_hot(label,1000))

    with train_summary_writer.as_default():
        tf.summary.scalar('loss', loss_value, step=i)
    
    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    i += 1
    #print(loss_value.numpy())
'''