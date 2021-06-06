#https://eremo2002.tistory.com/76

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add




input_tensor = Input(shape=(224, 224, 1), dtype='float32', name='input')

def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x) # "same" padsize is 5.. 
    x = Conv2D(64, (7, 7), stride=(2, 2))
    x = BatchNormalization()(x)
    x = Axtivation('relu')(x)
    x = Zeropadding2D(padding = (1, 1)) # why?
    
    return x

def conv2_layer(x):
    x = MaxPooling2D((3, 3), 2)(x)
    
    shortcut = x
    for i in range(3):
        if not i:
            x = Conv2D(64, (1, 1), stride = (1, 1))(x)
            