import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D


class ResBlock(Layer):

    def __init__(self, out_channel = 1, pooling = 'max'):
        super(ResBlock, self).__init__()
        self.C = out_channel
        self.pooling = pooling
        
    def build(self, input_shape):
        
        self.LN_1 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_1 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        self.LN_2 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_2 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        
        self.LN_3 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_3 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        self.LN_4 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_4 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        
        self.LN_5 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_5 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        self.LN_6 = tf.keras.layers.LayerNormalization(axis = [1, 2, 3], epsilon=1e-8)
        self.conv_6 = Conv2D(self.C, (3, 3), strides = (1, 1), padding='same', activation = None)
        
        self.gelu = tf.keras.activations.gelu
        
        if self.pooling == 'max':
            self.pool  = tf.keras.layers.MaxPool2D(
                pool_size=(3, 3), strides=(2, 2), padding='same')
        elif self.pooling == 'avg':
            self.pool = tf.keras.layers.AveragePooling2D(
                pool_size=(3, 3), strides=(2, 2), padding='same')
        else:
            raise ValueError("Invalid pooling")
    def call(self, x):
        
        x = tf.tile(x,[1,1,1,2])
        _x = self.conv_1(self.gelu(self.LN_1(x)))
        _x = self.conv_2(self.gelu(self.LN_2(_x)))
        _x = _x + x
        
        x = _x
        _x = self.conv_3(self.gelu(self.LN_3(x)))
        _x = self.conv_4(self.gelu(self.LN_4(_x)))
        _x = _x + x
        
        x = _x
        _x = self.conv_5(self.gelu(self.LN_5(x)))
        _x = self.conv_6(self.gelu(self.LN_6(_x)))
        _x = _x + x
        
        x = self.pool(_x)
        
        return x