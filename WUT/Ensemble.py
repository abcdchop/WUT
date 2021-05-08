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

class Ensemble():
    def __init__(self, Model, num_ens=3):
        
        """Ensemble Initializer. Turns a neural network into an ensemble of networks.

        Args:
            Model: Input Keras Model.
            num_ens: How many copies in the ensemble

        Returns:
            Nothing lol

        """
        
        self.ensemble = [Model() for _ in range(num_ens)]

    def compile(self, *args, **kwargs):
        
        """compile. Literally use this as you'd use the normal compile.

        Returns:
            Nothing lol

        """

        for submodel in self.ensemble:
            submodel.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        
        """fit. Literally use this as you'd use the normal fit.
        
        Returns:
            Nothing lol

        """

        for submodel in self.ensemble:
            submodel.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        
        """evaluate. Literally use this as you'd use the normal evaluate.

        Returns:
            the mean score of the ensemble

        """

        results = []
        for submodel in self.ensemble:
            test_scores = submodel.evaluate(*args, **kwargs)
            results.append(test_scores)
        if type(results[0]) is tuple:
            return list(zip(*results))
        return

    def predict(self, *args, return_std = True, **kwargs):
        """evaluate. Literally use this as you'd use the normal evaluate.

        Args:
            return_std: defaults to true, just checking if you actually want the std.

        Returns:
            a mean and a variance for each input as a N_testx2 matrix

        """        
        
        predictions = [submodel.predict(*args, **kwargs) for submodel in self.ensemble]
        predictions = tf.stack(predictions)

        mean_preds = tf.reduce_mean(predictions, axis = 0)

        if not return_std:
            return mean_preds

        mean_preds = tf.expand_dims(mean_preds, 1)
        std_preds = tf.math.reduce_std(predictions, axis = 0)
        std_preds = tf.expand_dims(std_preds, 1)

        return tf.concat([mean_preds, std_preds], axis = 1)

    def sample(self, *args, **kwargs):
        """evaluate. Literally use this as you'd use the normal evaluate.

        Args:
            return_std: Defaults to true, just checking if you actually want the std.

        Returns:
            Output of each network for each input (Number of ensembels x Number of test inputs x Number of outputs)

        """

        predictions = [submodel.predict(*args, **kwargs) for submodel in self.ensemble]
        predictions = tf.stack(predictions)

        return predictions