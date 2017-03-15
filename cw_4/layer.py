import tensorflow as tf
import os
DEFAULT_PADDING = 'SAME'


def flatten(inputs):
    layer_shape = inputs.get_shape()
    num_features = layer_shape[1:4].num_elements()
    return tf.reshape(inputs, [-1, num_features]), num_features


class Layer(object):
    def __init__(self, name):
        self.name = name
        self.inputs = None
        self.outputs = None

    def flatten(self):
        layer_shape = self.inputs.get_shape()
        num_features = layer_shape[1:4].num_elements()
        self.inputs = tf.reshape(self.inputs, [-1, num_features])


class Conv2dLayer(Layer):
    def __init__(self, inputs, weights, biases, name):
        super(Conv2dLayer, self).__init__(name)
        self.weights = weights
        self.biases = biases
        self.inputs = inputs
        self.outputs = None
        if inputs:
            self.set_inputs(inputs)

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):
            _conv = tf.nn.conv2d(self.inputs, variable(self.weights), [1, 1, 1, 1], padding=DEFAULT_PADDING)
            _pre = tf.nn.bias_add(_conv, _biases(self.biases))
            self.outputs = tf.nn.relu(_pre)


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
                self.outputs = tf.nn.relu(tf.matmul(self.inputs, weights) + biases)


class PoolLayer(Layer):
    def __init__(self, inputs, name):
        super(PoolLayer, self).__init__(name)
        self.inputs = inputs
        self.outputs = None

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


def max_pool(inputs, name='pool'):
    """Layer outputting the maximum of non-overlapping 1D pools of inputs."""
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                          padding=DEFAULT_PADDING, name=name)


def variable(shape, stddev=0.1, name='weights'):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))


def _biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.0, shape=shape, name=name))


def _flatten(inputs):
    layer_shape = inputs.get_shape()
    num_features = layer_shape[1:4].num_elements()
    return tf.reshape(inputs, [-1, num_features]), num_features


def conv2d(inputs, weights, name='conv2d'):
    with tf.name_scope(name):
        kernel = variable(weights or [5, 5, 3, 4])
        biases = _biases([4])
        _conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding=DEFAULT_PADDING)
        _pre = tf.nn.bias_add(_conv, biases)
        return tf.nn.relu(_pre)


def fully_connected_layer(inputs, input_dim, output_dim, final=False,
                          name='fc-layer'):
    with tf.name_scope(name):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_dim, output_dim],
                stddev=2. / ( input_dim + output_dim) ** 0.5),
            name='weights'
        )
        biases = tf.Variable(tf.zeros([output_dim]), name='biases')
        if final:
            outputs = tf.add(tf.matmul(inputs, weights), biases)
        else:
            outputs = tf.nn.relu(tf.matmul(inputs, weights) + biases)
        return outputs


def relu(inputs, name='relu'):
    """Layer implementing an element-wise rectified linear transformation."""
    return tf.nn.relu(inputs, name=name)


def graph_summary(error, accuracy, name, graph):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'train'), graph=graph)
    valid_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'valid'), graph=graph)
    return summary_op, train_writer, valid_writer
