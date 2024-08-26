import tensorflow as tf

##### Building Blocks #####
class Coefficients(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name=f'Coefficients')
        self.w_initializer = tf.keras.initializers.RandomUniform(minval=-1, maxval=1)
       
    def build(self, input_shape):
        self.w = self.add_weight(name='w', shape=[input_shape[-1] + 1, 1], initializer=self.w_initializer, trainable=True)

    def call(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.concat([tf.ones([x.shape[0],1], dtype= tf.float32),x], axis = -1)        
        return x @ self.w
    
    def __repr__(self):
        return f'Values: {self.w.numpy()}'
    
##### Model #####
class MultipleLinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__(name='LinearRegression')
        self.coeff = Coefficients()
        
    def __call__(self, x):
        return self.coeff(x)
    
    def compile(self, optimizer, y_mean = 0):
        super(MultipleLinearRegression, self).compile()
        self.optimizer = optimizer
        self.y_mean = tf.convert_to_tensor(y_mean, dtype=tf.float32)
    
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
        y = tf.convert_to_tensor(y, dtype=tf.float32)

        ### Forward pass
        with tf.GradientTape() as tape:
            y_hat = self(x)
            self.loss = self._RSS(y, y_hat)

        ### Metrics
        TSS = self._TSS(y_hat)
        self.accuracy = self._Rsquared(self.loss, TSS)
        
        ### Backward pass
        model_gradient = tape.gradient(self.loss, self.trainable_variables)

        try:
            self.optimizer.apply_gradients(zip(model_gradient, self.trainable_variables))
        except Exception as e:
            print(f"Error applying gradients: {e}")
            
        return {"Loss": self.loss, 'Accuracy': self.accuracy}