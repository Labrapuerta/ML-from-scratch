import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import Model

### Building blocks for the transformer model
class positional_encoding(Layer):
    def __init__(self, **kwargs):
        super(positional_encoding, self).__init__(**kwargs)

    def build(self, input_shape):
        positions = tf.range(input_shape[1], dtype=tf.float32)[:, tf.newaxis]
        dim = tf.range(input_shape[-1], dtype=tf.float32)[tf.newaxis, :]
        self.pe = positions / tf.pow(10000, 2 * (dim // 2) / input_shape[-1])
        self.pe = tf.where( tf.cast(dim % 2, tf.bool),tf.cos(self.pe), tf.sin(self.pe))

    def call(self, x):
        return self.pe + x

class multi_head_attention(Layer):
    def __init__(self, heads=8):
        super(multi_head_attention, self).__init__()
        self.Q_initializer = GlorotUniform()
        self.K_initializer = GlorotUniform()
        self.V_initializer = GlorotUniform()
        self.WO_initializer = GlorotUniform()
        self.heads = heads
    
    def build(self, input_shape):
        self.n_dims = input_shape[-1]
        self.WQ = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.Q_initializer, trainable=True)
        self.WK = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.K_initializer, trainable=True)
        self.WV = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.V_initializer, trainable=True)
        self.WO = self.add_weight(shape=(self.n_dims, self.n_dims), initializer= self.WO_initializer, trainable=True)
    
    def call(self,x):
        ### x shape = (batch_size, seq_len, n_dims)
        Q = x @ self.WQ
        K = x @ self.WK
        V = x @ self.WV

        ### Splitting the heads and stacking them
        Q = tf.stack(tf.split(Q, self.heads, axis=2))
        K = tf.stack(tf.split(K, self.heads, axis=2))
        V = tf.stack(tf.split(V, self.heads, axis=2))
        ### Applying the attention
        return self.attention(Q, K, V) @ self.WO

    def attention(self, Q, K, V):
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = softmax(QK / np.sqrt(self.n_dims), axis=-1)
        QKV = tf.matmul(QK, V)
        return tf.concat(tf.unstack(QKV, axis=0), axis=-1)

class add_n_norm(Layer):
    def __init__(self, epsilon=1e-6):
        super(add_n_norm, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.n_dims = input_shape[-1]
        self.gamma = self.add_weight(shape=(self.n_dims,), initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(self.n_dims,), initializer='zeros', trainable=True)
    
    def call(self, x, x_i):
        ## x_i is the input to the sublayer
        ## x is the output from the sublayer
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        std = tf.math.reduce_std(x, axis=-1, keepdims=True)
        normalized_values = (x - mean) / tf.sqrt(tf.square(std) + self.epsilon) * self.gamma + self.beta
        return normalized_values + x_i

class dense_layer(Layer):
    def __init__(self, n_out, activation):
        super().__init__(name=f'Neuron')
        self.w_initializer = GlorotUniform()
        self.b_initializer = GlorotUniform()
        self.n_out = n_out
        self.activation = activation
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=[input_shape[-1],self.n_out], initializer=self.w_initializer, trainable=True)
        self.b = self.add_weight(shape=[self.n_out,], initializer=self.b_initializer, trainable=True)
    
    def call(self, x):
        z = x @ self.w + self.b
        if self.activation:
            z =  self.activation(z)
        return z     

class masked_multi_head_attention(Layer):
    def __init__(self, heads=8):
        super(masked_multi_head_attention, self).__init__()
        self.Q_initializer = GlorotUniform()
        self.K_initializer = GlorotUniform()
        self.V_initializer = GlorotUniform()
        self.WO_initializer = GlorotUniform()
        self.heads = heads
    
    def build(self, input_shape):
        self.n_dims = input_shape[-1]
        self.WQ = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.Q_initializer, trainable=True)
        self.WK = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.K_initializer, trainable=True)
        self.WV = self.add_weight(shape=(self.n_dims, self.n_dims), initializer=self.V_initializer, trainable=True)
        self.WO = self.add_weight(shape=(self.n_dims, self.n_dims), initializer= self.WO_initializer, trainable=True)
        a = tf.linalg.band_part(tf.ones(shape= (input_shape[1], input_shape[1])), -1, 0)
        mask = tf.not_equal(a, 1)
        a = tf.where(mask, np.inf * -1, a)
        mask = tf.greater(a, 0)
        self.a = tf.where(mask, 0, a)

    def call(self,x,Q):
        ### x shape = (batch_size, seq_len, n_dims)
        Q = Q @ self.WQ
        K = x @ self.WK
        V = x @ self.WV
        
        ### Splitting the heads and stacking them
        Q = tf.stack(tf.split(Q, self.heads, axis=2))
        K = tf.stack(tf.split(K, self.heads, axis=2))
        V = tf.stack(tf.split(V, self.heads, axis=2))
        ### Applying the attention
        return self.attention(Q, K, V) @ self.WO

    def attention(self, Q, K, V):
        QK = tf.matmul(Q, K, transpose_b=True)
        QK = QK + self.a   ### Masking the attention
        QK = softmax(QK / np.sqrt(self.n_dims), axis=-1)
        QKV = tf.matmul(QK, V)
        return tf.concat(tf.unstack(QKV, axis=0), axis=-1)

class linear(Layer):
    def __init__(self, n_out):
        super(linear, self).__init__()
        self.n_out = n_out
    
    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.n_out), initializer='ones', trainable=True)

    def call(self, x):
        return softmax(x @ self.w, axis=-1)

### Encoder layer
class encoder_layer(Layer):
    def __init__(self, n_heads = 8):
        super(encoder_layer, self).__init__()
        self.mha = multi_head_attention(n_heads)
        self.add_norm1 = add_n_norm()
        self.dense = dense_layer(2048, tf.nn.relu)
        self.dense1 = dense_layer(512, None)
        self.add_norm2 = add_n_norm()
    
    def call(self, x):
        x1 = self.mha(x)
        x = self.add_norm1(x1, x)
        x1 = self.dense(x)
        x1 = self.dense1(x1)
        x = self.add_norm2(x, x1)
        return x   

### Decoder layer
class decoder_layer(Layer):
    def __init__(self, n_heads = 8):
        super(decoder_layer, self).__init__()
        self.mha1 = masked_multi_head_attention(n_heads)
        self.add_norm1 = add_n_norm()
        self.mha2 = multi_head_attention(n_heads)
        self.add_norm2 = add_n_norm()
        self.dense = dense_layer(2048, tf.nn.relu)
        self.dense1 = dense_layer(512, None)
        self.add_norm3 = add_n_norm()

    
    def call(self, x, enc):
        x1 = self.mha1(x, enc)
        x = self.add_norm1(x1, x)
        x1 = self.mha2(x)
        x = self.add_norm2(x, x1)
        x1 = self.dense(x)
        x1 = self.dense1(x1)
        x = self.add_norm3(x, x1)
        return x

#### Encoder
class encoder(Layer):
    def __init__(self, n_layers=6, n_heads=8, input_dims = 6 ,output_dims=512):
        super(encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dims, output_dims)
        self.pe = positional_encoding()
        self.layers = [encoder_layer(n_heads= n_heads) for _ in range(n_layers)]
    
    def call(self, x):
        x = self.embedding(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x)
        return x


### Decoder
class decoder(Layer):
    def __init__(self, n_layers=6, n_heads=8, output_dims=512, input_dims=6):
        super(decoder, self).__init__()
        self.transform_input = dense_layer(output_dims, None)
        self.pe = positional_encoding()
        self.layers = [decoder_layer(n_heads= n_heads) for _ in range(n_layers)]
        self.dense = dense_layer(512, tf.nn.relu)
        self.dense1 = dense_layer(256, None)
        self.dense2 = dense_layer(input_dims, None)


    def call(self, x, enc):
        x = self.transform_input(x)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc)
        x = self.dense(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return x

### Transformer model