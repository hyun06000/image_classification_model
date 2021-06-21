#https://eremo2002.tistory.com/76

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, Dense, GlobalAveragePooling2D, MaxPooling2D, ZeroPadding2D, Add




USE_BIAS = True

EPSILON = 1e-3

KERNEL_REGULARIZER = tf.keras.regularizers.l2(0.0001)
BIAS_REGULARIZER   = tf.keras.regularizers.l2(0.0)
BETA_REGULARIZER   = tf.keras.regularizers.l2(0.0)
GAMMA_REGULARIZER  = tf.keras.regularizers.l2(0.0)

input_tensor  = [Input(shape=(32, 32, 3), dtype = tf.float32, name='input')]
input_tensor += [Input(shape=(1), dtype = tf.bool, name='is_training')]


def per_pixel_mean_sub(image):
    pixel_mean = tf.math.reduce_mean(image, axis=None, keepdims=True)
    
    return image - pixel_mean


def conv1_layer(x):    
    
    x, is_training = x
    
    x = Conv2D(64, (3, 3), strides=(2, 2), padding = 'same', use_bias=USE_BIAS,
               kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
    x = BatchNormalization(
        epsilon=EPSILON,
        beta_regularizer=BETA_REGULARIZER,
        gamma_regularizer=GAMMA_REGULARIZER
    )(x, training=is_training)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
 
    return [x,is_training]


def conv2_layer(x):
    
    x, is_training = x
    
    x = MaxPooling2D((3, 3), 2)(x)     
    shortcut = x
    
    for i in range(3):
        if (i == 0):
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            shortcut = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(shortcut)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            shortcut = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(shortcut, training=is_training)
 
            x = Add()([x, shortcut])
            x = Activation('relu')(x)
            
            shortcut = x
 
        else:
            x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
            
            x = Conv2D(64, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
 
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)            
 
            x = Add()([x, shortcut])   
            x = Activation('relu')(x)  
 
            shortcut = x        
    
    return [x,is_training]
 
 
 
def conv3_layer(x):        
    x, is_training = x
    shortcut = x    
    
    for i in range(4):     
        if(i == 0):            
            x = Conv2D(128, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)        
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)  
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            shortcut = Conv2D(512, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(shortcut)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            shortcut = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(shortcut, training=is_training)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)    
 
            shortcut = x              
        
        else:
            x = Conv2D(128, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
            
            x = Conv2D(128, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
 
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)            
 
            x = Add()([x, shortcut])     
            x = Activation('relu')(x)
 
            shortcut = x      
            
    return [x,is_training]
 
 
 
def conv4_layer(x):
    x, is_training = x
    shortcut = x        
  
    for i in range(6):     
        if(i == 0):            
            x = Conv2D(256, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)        
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)  
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            shortcut = Conv2D(1024, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(shortcut)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            shortcut = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(shortcut, training=is_training)
 
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)
 
            shortcut = x               
        
        else:
            x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
            
            x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
 
            x = Conv2D(1024, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)            
 
            x = Add()([x, shortcut])    
            x = Activation('relu')(x)
 
            shortcut = x      
 
    return [x,is_training]
 
 
 
def conv5_layer(x):
    x, is_training = x
    shortcut = x    
  
    for i in range(3):     
        if(i == 0):            
            x = Conv2D(512, (1, 1), strides=(2, 2), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)        
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)  
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            shortcut = Conv2D(2048, (1, 1), strides=(2, 2), padding='valid',
                              use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(shortcut)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            shortcut = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(shortcut, training=is_training)            
 
            x = Add()([x, shortcut])  
            x = Activation('relu')(x)      
 
            shortcut = x               
        
        else:
            x = Conv2D(512, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
            
            x = Conv2D(512, (3, 3), strides=(1, 1), padding='same',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)
            x = Activation('relu')(x)
 
            x = Conv2D(2048, (1, 1), strides=(1, 1), padding='valid',
                       use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
            x = BatchNormalization(
                epsilon=EPSILON,
                beta_regularizer=BETA_REGULARIZER,
                gamma_regularizer=GAMMA_REGULARIZER
            )(x, training=is_training)           
            
            x = Add()([x, shortcut]) 
            x = Activation('relu')(x)       
 
            shortcut = x                  
 
    return [x,is_training]
 
 
x = conv1_layer(input_tensor)
x = conv2_layer(x)
x = conv3_layer(x)
x = conv4_layer(x)
x = conv5_layer(x)[0]
 
x = GlobalAveragePooling2D()(x)
output_tensor = Dense(10, activation='softmax',
                      use_bias=USE_BIAS, kernel_regularizer=KERNEL_REGULARIZER,
               bias_regularizer=BIAS_REGULARIZER)(x)
 
resnet50 = Model(input_tensor, output_tensor)
resnet50.summary()