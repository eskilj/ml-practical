# Working example for my blog post at:
# https://danijar.github.io/structuring-your-tensorflow-models
import tensorflow as tf


class Model:

    def __init__(self, name, layers):
        self.name = name
        self.layers = layers

    def get_layers(self, inputs):
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer.set_inputs(inputs)
            else:
                layer.set_inputs(self.layers[i - 1].outputs)

        return self.layers[-1]
