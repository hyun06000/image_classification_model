#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import time
from math import ceil

import tensorflow as tf
import tensorflow_datasets as tfds


# In[2]:


ds_name = 'imagenet2012'
new_size = 128

ds = tfds.load(ds_name)

def im_resize(im_tensor, new_size = 128):
    H, W, C = im_tensor.shape
    if H > W: 
        new_H, new_W = new_size, int(W*(new_size/H))
    elif W > H : 
        new_H, new_W = int(H*(new_size/W)), new_size
    else : 
        new_H, new_W = new_size, new_size
    re_im = tf.image.resize(im_tensor, [new_H, new_W])
    re_im = tf.image.resize_with_pad(re_im, new_size, new_size)
    re_im = tf.cast(re_im, dtype=tf.uint8)
    return re_im


# In[5]:

files_per_tfr = 256

#TFR version
for data_set in ['train','validation']:
    print('TO MAKE {}_set is STARTING'.format(data_set))
    
    tic = time.time()
    _ds = ds[data_set]#.take(256)
    num_of_images = len(_ds)
    num_of_tfr = ceil(num_of_images / files_per_tfr)
    print('num_of_images ::: ', num_of_images)
    print('num_of_tfr ::: ',num_of_tfr)
    cur_files, indx = 0, 0
    for example in _ds:
        im_tensor = example['image']
        label = example['label']

        im_tensor = im_resize(im_tensor, new_size)
        x = tf.io.serialize_tensor(im_tensor).numpy()
        y = label.numpy()
        
        if not cur_files:
            file_writer=\
            tf.io.TFRecordWriter(
                './data/TFRs/{2}/{2}.tfrecord-{0:05d}-of-{1:05d}'.format(indx,
                                                                         num_of_tfr,
                                                                         data_set)
                )
        
        record_bytes = \
        tf.train.Example(features=tf.train.Features(feature={
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[x])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(y)])),
        })).SerializeToString()
        
        file_writer.write(record_bytes)
        
        if cur_files == files_per_tfr - 1:
            file_writer.close()
            file_writer = None
        
        if indx == 0:
            toc = time.time()
            print('first iteration ::: ',toc-tic, ' s')
            print('total process will take {} s.'.format((toc-tic) * num_of_tfr))

        
        
        indx += 1
        cur_files = indx % files_per_tfr
        
    if file_writer:
        file_writer.close()
    toc = time.time()
    print(data_set,' ::: ',toc-tic)






# npy version
'''
for data_set in ['train','validation']:

    _ds = ds[data_set].take(100)
    indx = 0
    for example in _ds:
        im_tensor = example['image']
        label = example['label']

        im_tensor = im_resize(im_tensor)

        path = './data/prepared/{0}/{1}'.format(data_set,label.numpy())
        os.makedirs(path, exist_ok=True)

        np.save(path+"/{}.npy".format(indx),im_tensor)
        print(indx,end = "\r")
        indx += 1
'''