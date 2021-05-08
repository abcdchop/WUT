# WUT? - Wrapper for Uncertainty in Tensorflow

![](docs/logo.png "logo")

* [Documentation](doc/WUT/index.html)
* [Tutorial](WUT_Guide.ipynb)
* [Developer]() 

# Features
Welcome to WUT?! This is a library for uncertainty quantification in deep neural networks implemented in TensorFlow/Keras. Keras/TensorFlow models can be wrapped in one of the WUT? classes in a single line of code. For instance,
```python
from WUT.Ensemble import Ensemble`
model = Ensemble(<keras_model>)
```

You can estimate both the mean and standard deviation for test inputs.
![](docs/index.png "output")

Available uncertainty quantification models are,
| Model | Type of Uncertainty | Compatible with |
| --- | --- | --- | --- |
| [Ensemble]() | Epistemic | Any network  |
| [MCDropout]() | Epistemic | Any network  |
| [Variance Netwoorks]() | Epistemic | Fully connected layers only  |
| [Stochastic Variational Inference]() | Epistemic | Fully connected layers only  |
| [Mixture of Gaussians/Mixture Density Networks]() | Aleatoric | Any network  |


## Requirements: 
   `pip install tensorflow==2.4.1` (should work for >= v.2.2)
   `pip install tensorflow-probability --upgrade`
   `python3 -m pip install keras-mdn-layer`
This library has a dependency on this repo [here](https://github.com/cpmpercussion/keras-mdn-layer). Shoutout to compercussion on github for making the mdn layer, which is extremely good.

## For developers: 
Documents are generated using pdoc3. To install pdoc3 `pip install pdoc3`. To update the ducumentation `pdoc --html WUT --output-dir doc`. You might need to delete the subfolder `WUT/doc/WUT`.

We welcome any contributions please visit [Pull request](https://help.github.com/articles/using-pull-requests/)

