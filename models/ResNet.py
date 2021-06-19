#https://eremo2002.tistory.com/76

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add




USE_BIAS = True
REGULARIZER = tf.keras.regularizers.l2(0.0001)

input_tensor = Input(shape=(32, 32, 3), dtype='float32', name='input')

def conv1_layer(x):    
    x = ZeroPadding2D(padding=(3, 3))(x)
    x = Conv2D(64, (7, 7), strides=(2, 2), use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return x


def conv2_layer(x):         
    x = MaxPooling2D((3, 3), 2)(x)     
 
    shortcut = x
 
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return x
 
 
 
def conv3_layer(x):        
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return x
 
 
 
def conv4_layer(x):
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return x
 
 
 
def conv5_layer(x):
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(shortcut)
            x = BatchNormalization()(x)
            shortcut = BatchNormalization()(shortcut)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
            x = BatchNormalization()(x)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return x
 
 
 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)
 
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax',
                      use_bias=USE_BIAS, kernel_regularizer=REGULARIZER)(x)
 
resnet50 = Model(input_tensor, output_tensor)
resnet50.summary()