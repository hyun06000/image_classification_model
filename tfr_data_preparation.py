#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

from data.imagenet_labels import imagenet_labels


# In[2]:


ds_name = 'imagenet2012'
ds = tfds.load(ds_name)
print(ds)


# In[3]:


def resize_image(example, crop_size = 224):
    image, label = example['image'], example['label']

    H, W, C = image.shape
    if H < crop_size:
        p = (crop_size - H)//2
        paddings = tf.constant([[p, p], [0, 0], [0,0]])
        image = tf.pad(image, paddings)
    else:
        up, down = H//2 - crop_size//2, H//2 + crop_size//2
        image = image[up: down, :, :]
    if W < crop_size:
        p = crop_size - W
        paddings = tf.constant([[0, 0], [p, p], [0,0]])
        image = tf.pad(image, paddings)
    else:
        left, right = W//2 - crop_size//2, W//2 + crop_size//2
        image = image[:,left: right , :]
    
    return image, label


# In[4]:


tfr_path = './data/TFRecord/'
crop_size = 224

tr_ds = ds['train']#.take(1)
with tf.io.TFRecordWriter(tfr_path + 'train.tfrecords') as file_writer:
    for example in tr_ds:
        image, label = resize_image(example, crop_size)
        
        record_bytes =         tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image\
                                                                    .numpy()\
                                                                    .flatten()\
                                                                    .tolist())),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        })).SerializeToString()
        file_writer.write(record_bytes)


val_ds = ds['validation']#.take(1)
with tf.io.TFRecordWriter(tfr_path + 'validation.tfrecords') as file_writer:
    for example in val_ds:
        image, label = resize_image(example, crop_size)
        
        record_bytes =         tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=image\
                                                                    .numpy()\
                                                                    .flatten()\
                                                                    .tolist())),
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        })).SerializeToString()
        file_writer.write(record_bytes)
