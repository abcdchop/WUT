import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import tensorflow as tf



class Kuleshov():
    def __init__(self, model, xcal, ycal, num_buckets=100):
        self.num_buckets = num_buckets
        self.model = model

        yhat = model.predict(xcal)
        yhat_mean, yhat_std = yhat[:,0], yhat[:,1]
        percentiles = self.extract_percentiles(yhat_mean, yhat_std, ycal)
        self.bucket_indices = self.create_buckets(percentiles)

    def percentile(self, x, p):
        approx_index = int(p*(len(self.bucket_indices)))
        norm_percentile = self.bucket_indices[approx_index]
        yhat = self.model.predict(x)
        yhat_mean, yhat_std = yhat[:,0], yhat[:,1]
        dist = tfd.Normal(loc=yhat_mean, scale=yhat_std)
        return dist.quantile(norm_percentile)

    def extract_percentiles(self, mean, std, true):
        dist = tfd.Normal(loc=mean, scale=std)
        return dist.cdf(true)

    def create_buckets(self, percentiles):
        percentiles = tf.sort(percentiles, axis=0, direction='ASCENDING')
        percentiles = [0.0] + [percentiles[i * int(len(percentiles)/(self.num_buckets-1))] for i in range(self.num_buckets-1)] + [1.0]
        return percentiles
