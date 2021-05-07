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



class Dropout():
    
    
    def __init__(self, Model, rate, dropout_layers = [tf.keras.layers.Dense]):
        
        
        """Dropout Initializer. Turns a neural network into an droput network.

        Args:
            Model: Input Keras Model.
            rate: What rate to these neurons drop out
            dropout_layers: Types of layers to apply dropout to. Defaults are safe, add to this at your own risk.

        Returns:
            Nothing lol

        """        
        
        self.model = Model()

        def adddropout(denselayer):
            def func(self, inputs, **kwargs):
                x = core_ops.dense(inputs, self.kernel, self.bias, self.activation, dtype=self._compute_dtype_object)
                return tf.nn.dropout(x, noise_shape=None, rate=rate)
            return func

        for layer in self.model.layers[:-1]:
            if layer.__class__ in dropout_layers:
                func = adddropout(layer)
                layer.call = types.MethodType(func, layer)

    def compile(self, *args, **kwargs):
        
        """compile. Literally use this as you'd use the normal compile.

        Returns:
            Nothing lol

        """

        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        
        """fit. Literally use this as you'd use the normal keras fit.

        Returns:
            Nothing lol.

        """

        
        self.model.fit(*args, **kwargs)

    def evaluate(self, *args, trials = 3, **kwargs):
        
        
        """evaluate. Literally use this as you'd use the normal keras evaluate.
        Args:
            trials: how many times to run the MCDropout network to get empirical variance. I would recommend higher than 3.

        Returns:
            The model's score, evaluated on whatever inputs you just fed it.

        """

        
        results = []
        for _ in range(trials):
            test_scores = self.model.evaluate(*args, **kwargs)
            results.append(test_scores)
        if type(results[0]) is tuple:
            return list(zip(*results))
        return results

    def predict(self, *args, trials = 3, return_std = True, **kwargs):
        
        """predict. Literally use this as you'd use the normal keras predict.
        Args:
            trials: how many times to run the network
            return_std: checks if you would even like the std.

        Returns:
            a mean and variance, for each input.

        """

        predictions = [self.model.predict(*args, **kwargs) for _ in range(trials)]
        predictions = tf.stack(predictions)
        mean_preds = tf.reduce_mean(predictions, axis = 0)

        if not return_std:
            return mean_preds

        std_preds = tf.math.reduce_std(predictions, axis = 0)
        mean_preds = tf.expand_dims(mean_preds, 1)

        std_preds = tf.expand_dims(std_preds, 1)


        return tf.concat([mean_preds, std_preds], axis = 1)


    def sample(self, *args, trials = 3, **kwargs):
        """evaluate. Literally use this as you'd use the normal evaluate.

        Args:
            return_std: Defaults to true, just checking if you actually want the std.

        Returns:
            Output of each network for each input (Number of ensembels x Number of test inputs x Number of outputs)

        """

        predictions = [self.model.predict(*args, **kwargs) for _ in range(trials)]
        predictions = tf.stack(predictions)

        return predictions
