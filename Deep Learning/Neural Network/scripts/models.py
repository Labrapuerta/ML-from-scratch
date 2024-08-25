import tensorflow as tf

class Neuron(tf.Module):
    def __init__(self, n_dim, n_layer, n_neuron, activation):
        super().__init__(name=f'neuron_{n_neuron}')
        w_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        b_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        self.n_layer = n_layer
        self.neuron_index = n_neuron
        self.n_dim = n_dim
        self.activation = activation
        with tf.name_scope(f"layer_{self.n_layer+1}") as scope:
            self.w = tf.Variable(w_initializer(shape=[n_dim,1], dtype=tf.float32), trainable=True, name = f'weights_{n_neuron+1}', import_scope=scope)
            self.b = tf.Variable(b_initializer(shape=[1,1], dtype=tf.float32), trainable=True, name = f'bias_{n_neuron+1}', import_scope=scope)
    
    def __repr__(self):
        return f'w: {self.w.numpy().item()},b: {self.b.numpy().item()}'

    def __call__(self, x):
        self.x = tf.convert_to_tensor(x, dtype=tf.float32)
        self.shape = self.x.shape
        if self.n_layer == 0 and self.n_dim != 1:
            if self.n_dim !=  self.x.shape[-1]:
                raise ValueError(f'Input dimension {self.x.shape[0]} does not match neuron dimension {self.n_dim}') 
            self.x = self.x[:,self.neuron_index]
            self.x = tf.reshape(self.x, [self.shape[0],1])
        z = self.x @ self.w + self.b
        if self.activation:
            self.out =  self.activation(z)
        return z
    
    def parameters(self):
        return [self.b, self.w]
       
class Layer(tf.Module):
    def __init__(self, n_in, n_out, n_layer, activation = tf.tanh):
        super().__init__(name= f'layer_{n_layer}')
        self.n_inputs = n_in
        self.n_neurons = n_out
        self.n_layer = n_layer
        self.activation = activation
        self.neurons = [Neuron(n_in, n_layer, i , activation) for i in range(n_out)]

    def __repr__(self):
        return f'layer : {self.name}, neurons: {self.n_neurons}'

    def __call__(self, x):
        z =  tf.stack([neuron(x) for neuron in self.neurons], axis= -1)
        return tf.reshape(z, [z.shape[0], z.shape[-1]])
    
class Model(tf.keras.Model):
    def __init__(self, n_in, n_outs:list, activation:list):
        super().__init__(name='model')
        self.inputs = n_in
        sz = [n_in] + n_outs
        self.layerss =  [Layer(sz[i], sz[i+1], n_layer = i, activation = activation[i]) for i in range(len(n_outs))]

    def compile(self, optimizer, metrics = {}, y_mean = 0):
        super(Model, self).compile()
        self.optimizer = optimizer
        self._metrics = metrics
        for k,_ in self._metrics.items():
            setattr(self, k, 0)
        self.y_mean = tf.convert_to_tensor(y_mean, dtype=tf.float32)
    
    def __repr__(self):
        return f'{[layer for layer in self.layerss]}'

    def __call__(self, x):
        for layer in self.layerss:
            x = layer(x)
        return x

        #### Loss function
    def _RSS(self, y, y_hat):
        return tf.reduce_sum(tf.square(y - y_hat), axis = 0)[0]
    
    def _TSS(self, y_hat):
        return tf.reduce_sum(tf.square(y_hat - self.y_mean), axis = 0)[0]
    
    #### Accuracy metric
    def _Rsquared(self, RSS, TSS):
        return 1 - (RSS / TSS)
        
    def train_step(self, data):
        x, y = data
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.reshape(x, [x.shape[0], self.inputs])  # Corrected assignment here
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y = tf.reshape(y, [y.shape[0], 1])

        with tf.GradientTape() as tape:
            y_hat = self(x)
            self.loss = self._RSS(y, y_hat)

        self.accuracy = self._Rsquared(self.loss, self._TSS(y_hat))
        
        # Update metrics
        for k, v in self._metrics.items():
            value = getattr(self, k)
            v.update_state(value)

        # Backward pass
        model_gradient = tape.gradient(self.loss, self.trainable_variables)
        
        try:
            self.optimizer.apply_gradients(zip(model_gradient, self.trainable_variables))
        except Exception as e:
            print(f"Error applying gradients: {e}")

        return {"loss": self._metrics['loss'].result(), "accuracy": self._metrics['accuracy'].result()}


    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.reshape(x, [x.shape[0], self.inputs])  # Corrected assignment here
        y_hat = self(x)
        return y_hat
        
    @property
    def metrics(self):
        return [v for _,v in self._metrics.items()]