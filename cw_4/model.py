# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf
from layer import Conv2dLayer, PoolLayer, AffineLayer


class Model:

    def __init__(self, name, layers, activation=tf.nn.relu, train_epochs=10, initial_lr=0.001):
        self.name = name or 'unnamed model'
        self.layers = layers
        self.train_epochs = train_epochs
        self.initial_lr = initial_lr
        self.activation = activation

    def get_layers(self, inputs):
        for i, layer in enumerate(self.layers):
            layer.set_activation(self.activation)
            if i == 0:
                layer.set_inputs(inputs)
            else:
                layer.set_inputs(self.layers[i - 1].outputs)

        return self.layers[-1].outputs

    @classmethod
    def trial(cls):
        layers = [
            Conv2dLayer([5, 5, 3, 4], [4], 'conv_1', True),
            PoolLayer('pool_1'),
            AffineLayer('fc_1', True),
            AffineLayer('output', final_layer=True)
        ]
        print('Using trail model.')
        return cls(name='trial', layers=layers)
