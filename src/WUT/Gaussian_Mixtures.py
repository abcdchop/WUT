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



class Gaussian_Mixtures():
    def __init__(self, Model, num_mixtures=1):
        
        """Gaussian_MIxtures Initializer. Turns a neural network into an GMN.

        Args:
            Model: Input Keras Model.
            num_mixtures: how many total gaussians would you like to fit the output space to.

        Returns:
            Nothing lol

        """

        
        self.model = Model()


        layer = self.model.layers[-1]

        self.output_dim = layer.units
        layer.output_dim = layer.units
        self.num_mix = num_mixtures
        layer.num_mix = num_mixtures
        with tf.name_scope('MDN'):
            layer.mdn_mus = layers.Dense(layer.num_mix * layer.output_dim, name='mdn_mus')  # mix*output vals, no activation
            layer.mdn_sigmas = layers.Dense(self.num_mix * self.output_dim, activation=self.elu_plus_one_plus_epsilon, name='mdn_sigmas')  # mix*output vals exp activation
            layer.mdn_pi = layers.Dense(self.num_mix, name='mdn_pi')  # mix vals, logits





        def build(self, input_shape):
            with tf.name_scope('mus'):
                self.mdn_mus.build(input_shape)
            with tf.name_scope('sigmas'):
                self.mdn_sigmas.build(input_shape)
            with tf.name_scope('pis'):
                self.mdn_pi.build(input_shape)

        def call_func(self, x):
            with tf.name_scope('MDN'):
                mdn_out = layers.concatenate([self.mdn_mus(x),
                                              self.mdn_sigmas(x),
                                              self.mdn_pi(x)],
                                             name='mdn_outputs')
            return mdn_out


        def compute_output_shape(self, input_shape):
            """Returns output shape, showing the number of mixture parameters."""
            return (input_shape[0], (2 * self.output_dim * self.num_mix) + self.num_mix)

        def get_config(self):
            config = {
                "output_dimension": self.output_dim,
                "num_mixtures": self.num_mix
            }
            base_config = super(Dense, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

        layer.build = types.MethodType(build, layer)
        layer.call = types.MethodType(call_func, layer)
        layer._trainable_weights = layer.mdn_mus.trainable_weights + layer.mdn_sigmas.trainable_weights + layer.mdn_pi.trainable_weights
        layer._non_trainable_weights = layer.mdn_mus.non_trainable_weights + layer.mdn_sigmas.non_trainable_weights + layer.mdn_pi.non_trainable_weights
        layer.compute_output_shape = types.MethodType(compute_output_shape, layer)
        layer.get_config = types.MethodType(get_config, layer)


    def elu_plus_one_plus_epsilon(self, x):
        """ELU activation with a very small addition to help prevent
        NaN in loss."""
        return tf.keras.backend.elu(x) + 1 + .00001

    def compile(self, *args, loss=None, **kwargs):
        
        """compile. Literally use this as you'd use the normal compile, but don't use your own loss functions, unless your really deep in this. Let the default one go

        Args:
            loss: if you really wanna make your own loss function

        Returns:
            Nothing lol

        """        
        
        if loss is None:
            loss = mdn.get_mixture_loss_func(self.output_dim, self.num_mix)
        kwargs['loss'] = loss
        self.model.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        
        """fit. Literally use this as you'd use the normal keras fit.

        Returns:
            Nothing lol.

        """

        self.model.fit(*args, **kwargs)

    def evaluate(self, *args, **kwargs):
        
        """evaluate. Literally use this as you'd use the normal keras evaluate.
        Args:

        Returns:
            The model's score, evaluated on whatever inputs you just fed it.

        """

        return self.model.evaluate( *args, **kwargs)


    def predict(self, *args, **kwargs):
        
        """predict. Literally use this as you'd use the normal keras predict.

        Returns:
            a probability distribution over outputs, for each input.

        """


        all_preds = self.model.predict(*args, **kwargs)
        return self.get_dist(all_preds)

    def get_dist(self, y_pred):
        """turns an output into a distribution. Literally use this as you'd use the normal keras predict.
        Args:
            y_pred: nn output

        Returns:
            a probability distribution over outputs, for each input.

        """



        
        num_mix = self.num_mix
        output_dim = self.output_dim
        y_pred = tf.reshape(y_pred, [-1, (2 * num_mix * output_dim) + num_mix], name='reshape_ypreds')
        out_mu, out_sigma, out_pi = tf.split(y_pred, num_or_size_splits=[num_mix * output_dim,
                                                                         num_mix * output_dim,
                                                                         num_mix],
                                             axis=1, name='mdn_coef_split')
        cat = tfd.Categorical(logits=out_pi)
        component_splits = [output_dim] * num_mix
        mus = tf.split(out_mu, num_or_size_splits=component_splits, axis=1)
        sigs = tf.split(out_sigma, num_or_size_splits=component_splits, axis=1)
        coll = [tfd.MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                in zip(mus, sigs)]
        return tfd.Mixture(cat=cat, components=coll)




