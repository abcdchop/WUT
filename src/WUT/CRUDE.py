
import tensorflow as tf


class CRUDE():
    def __init__(self, model, xcal, ycal):
        yhat = model.predict(xcal)
        yhat_mean, yhat_std = yhat[:,0], yhat[:,1]
        true_error = yhat_mean - ycal
        scaled_true_error = true_error/yhat_std
        self.scaled_true_error = tf.sort(scaled_true_error, axis=0, direction='ASCENDING')
        self.len_cal = len(self.scaled_true_error)
        self.model = model

    def percentile(self, x, p):

        yhat = self.model.predict(x)
        yhat_mean, yhat_std = yhat[:,0], yhat[:,1]
        index = int(self.len_cal * p)
        error = self.scaled_true_error[index]
        scaled_error = yhat_std * error
        applied_error = yhat_mean + scaled_error

        return applied_error
