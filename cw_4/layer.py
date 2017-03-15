import tensorflow as tf
import os
DEFAULT_PADDING = 'SAME'


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.inputs = None
        self.outputs = None
        self.activation = tf.nn.relu

    def flatten(self):
        layer_shape = self.inputs.get_shape()
        num_features = layer_shape[1:4].num_elements()
        self.inputs = tf.reshape(self.inputs, [-1, num_features])

    def set_activation(self, activation):
        self.activation = activation


class Conv2dLayer(Layer):
    def __init__(self, weights, biases, name, apply_batch_norm=False):
        super(Conv2dLayer, self).__init__(name)
        self.weights = weights
        self.biases = biases
        self.apply_batch_norm = apply_batch_norm

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):
            _conv = tf.nn.conv2d(self.inputs, variable(self.weights), [1, 1, 1, 1], padding=DEFAULT_PADDING)
            _pre = tf.nn.bias_add(_conv, _biases(self.biases))
            if self.apply_batch_norm:
                _pre = _bn(_pre, _pre.get_shape()[-1].value)
            self.outputs = self.activation(_pre)


class AffineLayer(Layer):
    def __init__(self, name, flatten_inputs=False, final_layer=False):
        super(AffineLayer, self).__init__(name)
        self.final_layer = final_layer
        self.flatten_inputs = flatten_inputs

    def set_inputs(self, inputs):
        self.inputs = inputs
        if self.flatten_inputs:
            self.flatten()
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):

            input_dim = self.inputs.get_shape()[1].value
            output_dim = 10 if self.final_layer else int(input_dim/2)

            weights = variable(
                [input_dim, output_dim],
                2. / (input_dim + output_dim) ** 0.5)
            biases = tf.Variable(tf.zeros([output_dim]), name='biases')

            if self.final_layer:
                self.outputs = tf.add(tf.matmul(self.inputs, weights), biases)
            else:
                self.outputs = self.activation(tf.matmul(self.inputs, weights) + biases)


class PoolLayer(Layer):
    """Layer of non-overlapping 1D pools of inputs."""
    def __init__(self, name):
        super(PoolLayer, self).__init__(name)

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):
            self.outputs = tf.nn.max_pool(
                self.inputs,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding=DEFAULT_PADDING,
                name=self.name)


def _bn(inputs, output_dim):
    """Batch normalization on convolutional layers."""
    beta = tf.Variable(tf.constant(0.0, shape=[output_dim]), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[output_dim]), name='gamma')
    batch_mean, batch_var = tf.nn.moments(inputs, [0], name='moments')
    epsilon = 1e-3
    return tf.nn.batch_normalization(
        inputs, batch_mean, batch_var, beta, gamma, epsilon, 'bn'
    )


def variable(shape, stddev=0.1, name='weights'):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))


def _biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.0, shape=shape, name=name))


def graph_summary(error, accuracy, name, graph):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'train'), graph=graph)
    valid_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'valid'), graph=graph)
    return summary_op, train_writer, valid_writer
