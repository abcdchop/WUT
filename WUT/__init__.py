"""
Welcome to WUT?! This is a library for uncertainty quantification in deep neural networks implemented in TensorFlow/Keras. The codebase can be found at https://github.com/abcdchop/WUT.

Using the library is straightforward. Keras/TensorFlow models can be wrapped in one of the WUT? classes in a single line of code. For instance,

`from WUT.Ensemble import Ensemble`

`model = Ensemble(<keras_model>)`

You can estimate both the mean and standard deviation for a test input.
.. image:: ../images/index.png

To estimate the epistemic uncertainty, currently, we have implemented Ensemble, MCDropout, Variance Networks, and Stochastic Variational Inference (SVI). To estimate the multimodal aleatoric uncertainty, we have implemented a mixture of Gaussians at the last layer (a Mixture Density Network).


Note for developers: Documents are generated using pdoc3. To install pdoc3 `pip install pdoc3`. To update the ducumentation `pdoc --html WUT --output-dir doc`. You might need to delete the subfolder `WUT/doc/WUT`.
"""
