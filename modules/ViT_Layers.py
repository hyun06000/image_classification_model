import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv1D, Conv2D, Reshape, Permute, Attention

class PatchCNN(Layer):

    def __init__(self):
        super(PatchCNN, self).__init__()
        pass
    
    def build(self, input_shape):
        
        self.conv1 = Conv2D( 16, (5, 5), strides = (1, 1), padding='same', activation = 'relu')
        self.conv2 = Conv2D( 32, (5, 5), strides = (2, 2), padding='same', activation = 'relu')
        self.conv3 = Conv2D( 64, (5, 5), strides = (1, 1), padding='same', activation = 'relu')
        self.conv4 = Conv2D(128, (5, 5), strides = (2, 2), padding='same', activation = 'relu')
        
        self.reshape    = Reshape((8 * 8, 128))
        self.transpose = Permute((2,1))
        
    def call(self, x):
        if x.dtype == 'uint8':
            x = tf.cast(x, dtype = tf.float32) * (1./255)
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.reshape(x)
        x = self.transpose(x)
        
        return x

class PatchEmbedding(Layer):

    def __init__(self, D, patch_size):
        super(PatchEmbedding, self).__init__()
        
        self.D = D
        self.patch_size = patch_size
        
    def build(self, input_shape):
        
        self.flatten = tf.keras.layers.Flatten()
        self.linear_proj = tf.keras.layers.Dense(
                                    units = self.D,
                                    activation=None,
                                    use_bias=False,
                                    kernel_regularizer = tf.keras.regularizers.l2(0.1))
        self.image_shape = input_shape

    def call(self, image):
        if image.dtype != 'float32':
            raise TypeError("Your inpur tensor is {}.".format(image.dtype))
        
        p = self.patch_size
        B, H, W, C = self.image_shape
        
        patch_list = []
        for p_i in range(0, H, p):
            for p_j in range(0, W, p):
                patch = image[:, p_i:p_i+p, p_j:p_j+p, :]
                patch_list.append(self.flatten(patch))
        patchs = tf.stack(patch_list, axis = 1) # [B, N, P * P * C]
        
        x_p_E = self.linear_proj(patchs)
        
        return x_p_E


class ClassToken(Layer):

    def __init__(self):
        super(ClassToken, self).__init__()
        pass
    
    def build(self, input_shape):
        init = tf.random.normal(shape=tf.TensorShape([1, 1, input_shape[2]]))
        class_token = tf.Variable(
                init,
                trainable=True,
                name="class_toke1n",
                dtype=tf.float32,
                )
        self.tiled_class_token = tf.tile(class_token, [input_shape[0],1,1])
        
    def call(self, inputs):
        x_class = tf.concat([self.tiled_class_token,inputs], axis=1)
        
        return x_class

    
class Epos(Layer):

    def __init__(self):
        super(Epos, self).__init__()
        pass
    
    def build(self, input_shape):
        init = tf.random.normal(shape=tf.TensorShape([1, input_shape[1], input_shape[2]]))
        E_pos = tf.Variable(
            init,
            trainable=True,
            name="E_pos",
            dtype=tf.float32,
            )
        self.tiled_E_pos = tf.tile(E_pos,[input_shape[0],1,1])
        
    def call(self, inputs):
        z_0 = inputs + self.tiled_E_pos
        
        return z_0



class MultiHead_SelfAttention(Layer):

    def __init__(self, head = 8):
        super(MultiHead_SelfAttention, self).__init__()
        self.h = head
    
    def build(self, input_shape):
        self.D = input_shape[2]
        
        self.linear_proj = tf.keras.layers.Dense(
                                    units = 3 * self.D //self.h,
                                    activation=None,
                                    use_bias=False,
                                    kernel_regularizer = tf.keras.regularizers.l2(0.1))
        
        self.softmax = tf.keras.layers.Softmax(axis=-1)
        
        self.MSA_proj = tf.keras.layers.Dense(
                                    units = self.D,
                                    activation=None,
                                    use_bias=False,
                                    kernel_regularizer = tf.keras.regularizers.l2(0.1))
        
    def call(self, inputs):
        QKV = self.linear_proj(inputs) # [B, N, 3 * d_k]
        
        d_k = self.D//self.h
        Query = QKV[:, :, 0 * d_k :1 * d_k] # [B, N, d_k]
        Key   = QKV[:, :, 1 * d_k :2 * d_k]
        Value = QKV[:, :, 2 * d_k :3 * d_k]
        
        SA_list = []
        for n in range(d_k):
            nQ = Query[:,:,n:n+1] # [B, N, 1]
            nV = Value[:,:,n:n+1] # [B, N, 1]
            nK   = Key[:,:,n:n+1] # [B, N, 1]
            
            SDP_attention = tf.linalg.matmul(nQ, nK, transpose_b=True) # [B, N, N]
            SDP_attention = SDP_attention / tf.math.sqrt(tf.cast(d_k, dtype=tf.float32)) # [B, N, N]
            SDP_attention = self.softmax(SDP_attention) # [B, N, N]
            SDP_attention = tf.linalg.matmul(SDP_attention, nV) # [B, N, 1]
            
            SA_list.append(SDP_attention)
        
        SA = tf.concat(SA_list, axis = -1) # [B, N, d_k]
        
        MSA = self.MSA_proj(SA) # [B, N, D]
        
        return MSA

    
class FeedForward(Layer):

    def __init__(self, hidden_size_D, last = False, num_of_class = 10):
        super(FeedForward, self).__init__()
        self.hidden_size_D = hidden_size_D
        self.last = last
        self.num_of_class = num_of_class
        
    def build(self, input_shape):
        
        D = self.hidden_size_D
        self.conv1 = Conv1D( D, 1, strides = 1, padding='same',
                            activation=tf.keras.activations.gelu,
                            kernel_regularizer = tf.keras.regularizers.l2(0.1))
        if self.last:
            self.conv2 = Conv1D(self.num_of_class, 1, strides = 1, padding='same',
                            activation = None,
                               kernel_regularizer = tf.keras.regularizers.l2(0.1))
        else:
            self.conv2 = Conv1D( D, 1, strides = 1, padding='same',
                                activation = None,
                               kernel_regularizer = tf.keras.regularizers.l2(0.1))
        
    def call(self, x):
        
        x = self.conv2(self.conv1(x))
        
        return x

class ViT_Block(Layer):

    def __init__(self, hidden_size_D = 1024, head = 8, last = False, num_of_class = 10):
        super(ViT_Block, self).__init__()
        self.h = head
        self.D = hidden_size_D
        self.last = last
        self.num_of_class = num_of_class
        
        
    def build(self, input_shape):
        
        self.LN_0 = tf.keras.layers.LayerNormalization(axis = -1)
        self.MSA = MultiHead_SelfAttention(head = self.h)
        
        self.LN_1 = tf.keras.layers.LayerNormalization(axis = -1)
        self.FF = FeedForward(hidden_size_D = self.D, 
                              last = self.last,
                              num_of_class = self.num_of_class)
        
    def call(self, x):
        
        x = self.MSA(self.LN_0(x)) + x
        
        if self.last:
            x = self.FF(self.LN_1(x))
        else:
            x = self.FF(self.LN_1(x)) + x
        
        return x