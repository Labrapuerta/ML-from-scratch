import tensorflow as tf

##### Building Blocks #####
class Coefficients(tf.Module):
    def __init__(self, n_dim):
        super().__init__(name=f'Coefficients')
        w_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
        self.n_dim = n_dim
        self.w = tf.Variable(w_initializer(shape=[n_dim + 1,1], dtype=tf.float32), trainable=True)
       
    def __repr__(self):
        return f'Values: {self.w.numpy()}'

    def __call__(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.concat([tf.ones_like(x, dtype= tf.float32),x], axis = -1)
        if self.n_dim + 1 != x.shape[-1]:
            raise ValueError(f'Expected input dimension: {self.n_dim}, got: {x.shape[-1] - 1}')
        return x @ self.w
       
##### Model #####
class LinearRegression(tf.keras.Model):
    def __init__(self, n_dim):
        super().__init__(name='LinearRegression')
        self.coeff = Coefficients(n_dim)
        
    def __call__(self, x):
        return self.coeff(x)
    
    def compile(self, optimizer, y_mean = 0, metrics = {}):
        super(LinearRegression, self).compile()
        self.optimizer = optimizer
        self.y_mean = tf.convert_to_tensor(y_mean, dtype=tf.float32)
        self._metrics = metrics
        for k,_ in self._metrics.items():
            setattr(self, k, 0)
    
    def train_step(self, data):
        x, y = data 
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        ### Forward pass
        with tf.GradientTape() as tape:
            y_hat = self(x)
            self.loss = tf.reduce_sum(tf.square(y - y_hat), axis = 0)[0]
        TSS = tf.reduce_sum(tf.square(y - self.y_mean), axis = 0)[0]
        self.accuracy = 1 - (self.loss / TSS)
        
        ### Metrics
        for k,v in self._metrics.items():
            value = getattr(self, k)
            v.update_state(value)

        ### Backward pass
        model_gradient = tape.gradient(self.loss, self.trainable_variables)

        try:
            self.optimizer.apply_gradients(zip(model_gradient, self.trainable_variables))
        except Exception as e:
            print(f"Error applying gradients: {e}")
            
        return {"Loss": self._metrics['loss'].result(), 'Accuracy': self._metrics['accuracy'].result()}


    def predict(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return self(x) + self._metrics['loss'].result()
    
        
    @property
    def metrics(self):
        return [v for _,v in self._metrics.items()]