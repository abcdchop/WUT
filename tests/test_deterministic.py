import numpy as np
import tensorflow as tf

def test_simple():
    assert (10 < 55)

def test_deterministic():
    # Groundtruth function
    def groundtruth1(X):
        return np.sin(12 * X)

    # Training dataset
    N = 10000
    Xreg1 = np.random.random((N, 1))
    Yreg1 = groundtruth1(Xreg1) + 0.05 * np.random.random((N, 1))
    print("X-Y train shapes:", Xreg1.shape, Yreg1.shape)
    ylims = [-1.5, 1.5]  # This will be used for plotting
    # plot1d_meanvar(Xreg1, Yreg1, Xreg1q, None, None, ylims)

    # Query dataset for prediction
    Nq = 1000
    Xreg1q = np.linspace(-0.2, 1.2, Nq)[:, None]
    print("X query/predict shapes:", Xreg1q.shape)

    # Test dataset for evaluation
    Xreg1t = Xreg1q
    Yreg1t = groundtruth1(Xreg1q)
    print("X-Y test shapes:", Xreg1t.shape, Yreg1t.shape)

    # Network
    class RegModel1(tf.keras.Model):
        def __init__(self):
            super(RegModel1, self).__init__()
            self.dense1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)
            self.dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
            self.dense3 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
            self.dense4 = tf.keras.layers.Dense(1)

        def call(self, inputs, training=False):
            x = self.dense1(inputs)
            x = self.dense2(x)
            x = self.dense3(x)
            return self.dense4(x)

    # Train
    regmodel1 = RegModel1()
    regmodel1.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MSE)
    regmodel1.fit(Xreg1, Yreg1, epochs=30, batch_size=20)

    # Predict
    Yreg1q = regmodel1.predict(Xreg1q)
    print("X-Y predict shapes:", Xreg1q.shape, Yreg1q.shape)
    # plot1d_meanvar(Xreg1, Yreg1, Xreg1q, tf.expand_dims(Yreg1q,2), None, ylims)

    # Evaluate
    print("Evaluate train:", regmodel1.evaluate(Xreg1, Yreg1))
    print("Evaluate test:", regmodel1.evaluate(Xreg1t, Yreg1t))

    assert (regmodel1.evaluate(Xreg1t, Yreg1t) >= 0)
