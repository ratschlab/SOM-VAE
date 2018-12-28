# SOM-VAE model

This repository contains a TensorFlow implementation of the self-organizing map variational autoencoder as described in the paper [SOM-VAE: Interpretable Discrete Representation Learning on Time Series](https://arxiv.org/abs/1806.02199).

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to install and run the model, you will need a working Python 3 distribution as well as a NVIDIA GPU with CUDA and cuDNN installed.

### Installing

In order to install the model and run it, you have to follow these steps:

* Clone the repository, i.e. run `git clone https://github.com/ratschlab/SOM-VAE`
* Change into the directory, i.e. run `cd SOM-VAE`
* Install the requirements, i.e. run `pip install -r requirements.txt`
* Install the package itself, i.e. run `pip install .`
* Change into the code directory, i.e. `cd som_vae`

Now you should be able to run the code, e.g. do `python somvae_train.py`.

### Training the model

The SOM-VAE model is defined in [somvae_model.py](som_vae/somvae_model.py).
The training script is [somvae_train.py](som_vae/somvae_train.py).

If you just want to train the model with default parameter settings, you can run

```
python somvae_train.py
```

This will download the MNIST data set into `data/MNIST_data/` and train on it. Afterwards, it will evaluate the trained model in terms of different clustering performance measures.

The parameters are handled using [sacred](https://github.com/IDSIA/sacred).
That means that if you want to run the model with a different parameter setting, e.g. a latent space dimensionality of 32, you can just call the training script like

```
python somvae_train.py with latent_dim=32
```

Per default, the script will generate time courses of linearly interpolated MNIST digits.
To train on normal MNIST instead, run

```
python somvae_train.py with time_series=False
```

Note that for non-time-series training, you should also set the loss parameters `gamma` and `tau` to 0.
If you want to save the model for later use, run

```
python somvae_train.py with save_model=True
```

If you want to train on [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) istead of normal MNIST, download the data set into `data/fashion/` and run

```
python somvae_train.py with data_set="fashion"
```

For more details regarding the different model parameters and how to set them, please look at the documentation in the [code](som_vae/somvae_train.py) and at the sacred documentation.

### Hyperparameter optimization

If you want to optimize the models hyperparameters, you have to additionally install [labwatch](https://github.com/automl/labwatch) and [SMAC](https://github.com/automl/SMAC3) and comment the commented out lines in [somvae_train.py](som_vae/somvae_train.py) in.
Note that you also have to run a local distribution of the [MongoDB](ihttps://www.mongodb.com/).

### Train on other kinds of data

If you want to train on other types of data, you have to run the training with

```
python somvae_train.py with mnist=False
```

Moreover, you have to define the correct dimensionality in the respective `input_length` and `input_channels` parameters of the model, provide a suitable data generator in [somvae_train.py](som_vae/somvae_train.py) and potentially change the dimensionality of the layers in [somvae_model.py](som_vae/somvae_model.py).

## Authors

* **Vincent Fortuin** - [ETH website](https://bmi.inf.ethz.ch/people/person/vincent-fortuin/)

See also the list of [contributors](https://github.com/ratschlab/SOM-VAE/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details

