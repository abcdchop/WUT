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


class Variance_Network():
    def __init__(self, Model, Std_Model = None, error_activation = None, one_hot = True):
        
        
        """Variance Network Initializer. Turns a neural network into Variance network.

        Args:
            Model: Input Keras Model.
            Std_Model: what model would you like to use to predict variance. If not specified, assumes default model. (We Thoughtfully remove the activation on your last layer for you.)
            error_activation: use this only if you don't specify Std_Model, it will change the activation function on the last layer of the Std model to this
            one_hot: if True, assumes you're feeding us one hot vectors as targets. If false, assumes its just labels, and we'll make it a one hot.

        Returns:
            Nothing lol

        """

        
        
        
        if Std_Model is None:
            Std_Model = Model
        self.model = Model()
        self.std_model = Std_Model()

        self.error_norm = 1
        self.error_activation = error_activation
        self.one_hot = one_hot
            



        layer = self.std_model.layers[-1]
        layer.activation = activations.get(error_activation)


    def compile(self, *args, Std_loss = None, **kwargs):
        
        """compile. Literally use this as you'd use the normal compile, but you can also specify a loss for the variance predictor. defaults to MSE.

        Args:
            Std_loss: loss function for Std_Model.

        Returns:
            Nothing lol

        """

        
        self.model.compile(*args, **kwargs)
        new_kwargs = kwargs
        if Std_loss == None:
            Std_loss = tf.keras.losses.MSE
        new_kwargs['loss'] = Std_loss
        self.std_model.compile(*args, **new_kwargs)



    def fit(self, *args, **kwargs):
        
        """fit. Literally use this as you'd use the normal keras fit.

        Returns:
            Nothing lol.

        """

        
        self.model.fit(*args, **kwargs)
        preds = self.model.predict(args[0])
        if self.one_hot == False:
            args = list(args)
            max_val = tf.reduce_max(args[1])
            max_val = tf.cast(max_val + 1, tf.int32)
            onehot = tf.one_hot(args[1], max_val)
            args[1] = onehot
            args = tuple(args)
        preds = preds.reshape(args[1].shape)
        errors = ((args[1] - preds)**2)**.5
        new_args = list(args)
        new_args[1] = tf.reshape(errors, args[1].shape)
        new_args = tuple(new_args)
        self.std_model.fit(*new_args, **kwargs)

    def evaluate(self, *args, **kwargs):
        
        """evaluate. Literally use this as you'd use the normal keras evaluate.

        Returns:
            Nothing lol.

        """

        
        return self.model.evaluate( *args, **kwargs)

    def predict(self, *args, return_std = True, **kwargs):
        
        """predict. Literally use this as you'd use the normal keras predict.
        Args:
            return_std: checks if you would even like the std.
            
        Returns:
            a mean and variance, for each input.

        """        
        
        mean_preds = self.model.predict(*args, **kwargs)
        std_preds = self.std_model.predict(*args, **kwargs)
        mean_preds = tf.expand_dims(mean_preds, 1)
        std_preds = (tf.expand_dims(std_preds, 1) * self.error_norm)

        if not return_std:
            return np.mean(tf.stack(predictions), 0)
        return tf.concat([mean_preds, std_preds], axis = 1)
