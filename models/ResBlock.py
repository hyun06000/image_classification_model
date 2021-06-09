#https://eremo2002.tistory.com/76

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add




input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')

def conv1_layer(x):
    x = ZeroPadding2D(padding=(3, 3))(x) # "same" padsize is 5.. 
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding = (1, 1))(x) # why?
    
    return x

def ResBlock(x, C, P = True):
    if P:
        x = MaxPooling2D((3, 3), 2)(x)
    
    shortcut = x
    for i in range(3):
        if not i:
            x = Conv2D(C, (1, 1), strides = (1, 1))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(C, (3, 3), strides = (1, 1), padding = 'same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(C * 4, (1, 1), strides = (1, 1))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            shortcut = Conv2D(C * 4, (1, 1), strides = (1, 1))(shortcut)
            shortcut = BatchNormalization()(shortcut)
            
            x = Add()([x,shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
        
        else:
            x = Conv2D(C, (1, 1), strides = (1, 1))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(C, (3, 3), strides = (1, 1), padding = 'same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(C * 4, (1, 1), strides = (1, 1))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Add()([x,shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
    return x

x = conv1_layer(input_tensor)
x = ResBlock(x,64, False)
x = ResBlock(x,128,False)
x = ResBlock(x,256,)
x = ResBlock(x,512,)
x = ResBlock(x,1024)

x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation = 'softmax')(x)

resnet = Model(input_tensor,output_tensor)