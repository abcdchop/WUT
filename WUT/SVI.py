import numpy as np
import tensorflow as tf
import types
from tensorflow.python.keras.layers.ops import core as core_ops
import mdn
from tensorflow.compat.v1.keras import layers
from tensorflow.python.keras import activations
from tensorflow_probability import distributions as tfd
from keras import backend as K
from keras import activations, initializers
import tensorflow_probability as tfp


class SVI():
    def __init__(self, Model, kl_weight=1., prior_sigma_1=1.5, prior_sigma_2=0.1, prior_pi=0.5, SVI_Layers=[tf.keras.layers.Dense],  normalize = True, task = 'regression', one_hot = True, **kwargs):

        """SVI Initializer. Turns a neural network into an SVI network.

        Args:
            Model: Input Keras Model.
            kl_weight: Weight Parameter for KL Divergence.
            prior_sigma_1: First sigma for prior on weights
            prior_sigma_2: Second sigma for prior on weights
            prior_pi: First pi for prior on weights, second is 1 - prior_pi
            SVI_Layers: Layer types for which SVI can be applied-- the defaults, dense are garunteed safe, add others at your own risk
            normalize: Whether to normalize input and output values before using-- if you get nans, trying switching! Regression only
            task: regression or classification
            one_hot: are the input targets in one hot format. True if they inputs are one hot. Classification only.

        Returns:
            Nothing lol

        """
                
        self.model = Model()
        self.SVI_Layers = SVI_Layers
        self.one_hot = one_hot
        self.use_normalization = normalize
        self.task = task
        
        self.train_std = 0                
        self.xmean, self.xstd = 0., 1.
        self.ymean, self.ystd = 0., 1.
        
        if task == 'regression':
            last_layer = self.model.layers[-1]
            self.dim = last_layer.units
            last_layer.units = 2 * last_layer.units   
        if task == 'classification':
            self.dim = self.model.layers[-1].units
            
        def compute_output_shape(self, input_shape):
            return input_shape[0], self.units

        def kl_loss(self, w, mu, sigma):
            variational_dist = tfp.distributions.Normal(mu, sigma)
            return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

        def build(self, input_shape):
            self.kernel_mu = self.add_weight(name='kernel_mu',
                                             shape=(input_shape[1], self.units),
                                             initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                             trainable=True)
            self.bias_mu = self.add_weight(name='bias_mu',
                                           shape=(self.units,),
                                           initializer=initializers.RandomNormal(stddev=self.init_sigma),
                                           trainable=True)
            self.kernel_rho = self.add_weight(name='kernel_rho',
                                              shape=(input_shape[1], self.units),
                                              initializer=initializers.Constant(0.0),
                                              trainable=True)
            self.bias_rho = self.add_weight(name='bias_rho',
                                            shape=(self.units,),
                                            initializer=initializers.Constant(0.0),
                                            trainable=True)
            self._trainable_weights = [self.kernel_mu, self.bias_mu, self.kernel_rho, self.bias_rho]# 
            
        def call(self, inputs, **kwargs):
            
            if self.built == False:
                self.build(inputs.shape)
                self.built = True
            
            kernel_sigma = tf.math.softplus(self.kernel_rho)
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

            bias_sigma = tf.math.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

            self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                          self.kl_loss(bias, self.bias_mu, bias_sigma))

            return self.activation(K.dot(inputs, kernel) + bias)

        def log_prior_prob(self, w):
            comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
            comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
            return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                         self.prior_pi_2 * comp_2_dist.prob(w))
        
        
        for layer in self.model.layers:
            if layer.__class__ in SVI_Layers: 
                layer.kl_weight = kl_weight
                layer.prior_sigma_1 = prior_sigma_1
                layer.prior_sigma_2 = prior_sigma_2
                layer.prior_pi_1 = prior_pi
                layer.prior_pi_2 = 1.0 - prior_pi
                layer.init_sigma = np.sqrt(layer.prior_pi_1 * layer.prior_sigma_1 ** 2 +
                                          layer.prior_pi_2 * layer.prior_sigma_2 ** 2)
                layer.compute_output_shape = types.MethodType(compute_output_shape, layer)
                layer.build = types.MethodType(build, layer)
                layer.call = types.MethodType(call, layer)
                layer.kl_loss = types.MethodType(kl_loss, layer)
                layer.log_prior_prob = types.MethodType(log_prior_prob, layer)
                layer.built = False
            
    def fit_normalize(self, X):
        """fit_normalize. helper function to normalize input values and generate future normalizations

        Returns:
            normalized X, mean, std

        """
        
        
        mean = tf.math.reduce_mean(X, axis=0, keepdims=True)
        std = tf.math.reduce_std(X, axis=0, keepdims=True)
        return (X - mean)/std, mean, std
    
    def unnormalize(self, Y):
        
        """unnormalize. helper function to unnormalize output values

        Returns:
            unnormalized Y

        """

        return (Y*self.ystd) + self.ymean
    
    def Y_normalize(self, Y):
        
        """Y_normalize. helper function to normalize output values in training set

        Returns:
            normalized Y

        """

        return (Y - self.ymean)/self.ystd
    
    def std_unnormalize(self, Y):
        """unnormalize. helper function to unnormalize output values by varying std-- used for std predictions.

        Returns:
            unnormalized Y

        """
        
        return Y * self.ystd
    
    def normalize(self, X):
        """normalize. helper function to normalize input values

        Returns:
            normalized X

        """

        return (X - self.xmean)/self.xstd
        
    def compile(self, *args, loss=None, **kwargs):
        
        """compile. Literally use this as you'd use the normal compile, but don't use your own loss functions, unless your really deep in this. Let the default one go

        Args:
            loss: if you really wanna make your own loss function

        Returns:
            Nothing lol

        """


        
        if loss is None:
            loss = self.neg_log_likelihood
        else:
            print("Warning: you might be in for a rocky ride here, you specified your own loss function. If you don't know what your doing, don't do this! Loss functions have to be in the style of neg_log_likelihood in the source code/")
        kwargs['loss'] = loss          
        self.model.compile(*args, **kwargs)   
        
        
    def elu_plus_one_plus_epsilon(self, x):
        """ELU activation with a very small addition to help prevent
        NaN in loss."""
        return tf.keras.backend.elu(x) + 1 + .00001

    def neg_log_likelihood(self, y_obs, y_pred):
        if self.task == 'regression':
            y_means, y_stds = tf.split(y_pred, [self.dim, self.dim], axis = 1)
            if self.train_std == 1:
                pass
                y_stds = (y_stds)**2. + .1
            else:
                y_stds = tf.constant(1.0)
            dist = tfp.distributions.Normal(loc=y_means, scale=y_stds )
            return K.sum(-dist.log_prob(tf.dtypes.cast(y_obs, tf.float32)))
        
        if self.task == 'classification':
            
            y_pred = (tf.reshape(y_pred, [-1, self.dim]))
            y_obs = tf.reshape(y_obs, [-1, self.dim, 1])
            dist = tfp.distributions.Categorical(logits = y_pred)
            return K.sum(-dist.log_prob(tf.dtypes.cast(y_obs, tf.int32)))




    def evaluate(self, *args, **kwargs):
        
        """evaluate. Literally use this as you'd use the normal keras evaluate.
        Returns:
            The model's score, evaluated on whatever inputs you just fed it.

        """

        
        if self.use_normalization is True:
            args = list(args)
            args[0] = self.normalize(args[0])
            if self.task == 'regression':
                args[1] = self.Y_normalize(args[1])
            args = tuple(args)
            
        if (self.one_hot == False) and self.task == 'classification':
            args = list(args)
            data_amnt = args[1].shape[0]
            args[1] = tf.one_hot(args[1] , tf.dtypes.cast(tf.reduce_max(args[1] ) + 1, tf.int32))
            args[1] = tf.reshape(args[1], [data_amnt, -1])
            args = tuple(args)

        return self.model.evaluate( *args, **kwargs)
    
    
    def fit(self, *args, **kwargs):
        
        """fit. Literally use this as you'd use the normal keras fit.

        Returns:
            Nothing lol.

        """

        
        
        if self.use_normalization is True:
            args = list(args)
            args[0], self.xmean, self.xstd = self.fit_normalize(args[0])
            if self.task == 'regression':
                args[1], self.ymean, self.ystd = self.fit_normalize(args[1])
            args = tuple(args)
        
        if 'batch_size' not in kwargs.keys():
            kwargs['batch_size'] = 42
        
        for layer in self.model.layers:
            if layer.__class__ in self.SVI_Layers:            
                layer.kl_weight = kwargs['batch_size'] * layer.kl_weight/args[0].shape[0]
                
        if (self.one_hot == False) and self.task == 'classification':
            args = list(args)
            data_amnt = args[1].shape[0]
            args[1] = tf.one_hot(args[1] , tf.dtypes.cast(tf.reduce_max(args[1] ) + 1, tf.int32))
            args[1] = tf.reshape(args[1], [data_amnt, -1])
            args = tuple(args)
            
        if 'validation_data' in kwargs.keys():
            kwargs['validation_data'] = list(kwargs['validation_data'])
            data_amnt = kwargs['validation_data'][1].shape[0]
            args[1] = tf.one_hot(kwargs['validation_data'][1] , tf.dtypes.cast(tf.reduce_max(kwargs['validation_data'][1] ) + 1, tf.int32))
            args[1] = tf.reshape(kwargs['validation_data'][1], [data_amnt, -1])
            args = tuple(args)

        self.model.fit(*args, **kwargs)
        
        if self.task == 'regression':
            self.train_std = 1
            self.model.fit(*args, **kwargs)
            self.train_std = 0

        for layer in self.model.layers:
            if layer.__class__ in self.SVI_Layers:            
                layer.kl_weight =  layer.kl_weight*args[0].shape[0]/kwargs['batch_size']

    def predict(self, *args, return_std = True, **kwargs):
        
        """predict. Literally use this as you'd use the normal keras predict.

        Returns:
            a probability distribution over outputs, for each input.

        """

        
        if self.use_normalization is True:
            args = list(args)
            args[0] = self.normalize(args[0])
            args = tuple(args)
        y_pred = self.model.predict(*args, **kwargs)
        
        if self.task == 'regression':
            y_means, y_stds = tf.split(y_pred, [self.dim, self.dim], axis = 1)
            if self.use_normalization is True:
                y_means = self.unnormalize(y_means)
                y_stds = self.std_unnormalize(tf.exp(y_stds))
            dist = tfp.distributions.Normal(loc=y_means, scale=y_stds)
            return dist
        
        if self.task == 'classification':
            y_pred = tf.reshape(y_pred,  [-1, self.dim])
            dist = tfp.distributions.Categorical(logits= y_pred)
            return dist