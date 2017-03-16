import tensorflow as tf
DEFAULT_PADDING = 'SAME'


class Layer(object):
    """
    General Layer class
    """
    def __init__(self, name):
        self.name = name
        self.inputs = None
        self.outputs = None
        self.activation = tf.nn.relu

    def set_inputs(self, inputs):
        """
        Sets input for Layer
        :param inputs: tf.Tensor
        """
        self.inputs = inputs
        self.set_outputs()

    def set_outputs(self):
        pass

    def flatten(self):
        """
        Flatten inputs of layer
        """
        layer_shape = self.inputs.get_shape()
        num_features = layer_shape[1:4].num_elements()
        self.inputs = tf.reshape(self.inputs, [-1, num_features])

    def set_activation(self, activation):
        """
        Set layer activation function
        :param activation:
        :return:
        """
        self.activation = activation


class Conv2dLayer(Layer):
    """
    Convolutional Layer
    """
    def __init__(self, weights, biases, name, apply_batch_norm=False, weight_decay=None):
        super(Conv2dLayer, self).__init__(name)
        self.weights = weights
        self.biases = biases
        self.apply_batch_norm = apply_batch_norm
        self.weight_decay = weight_decay

    def set_outputs(self):
        with tf.name_scope(self.name):

            self.biases = _biases(self.biases)
            self.weights = _weights(self.weights, decay=self.weight_decay)

            _conv = tf.nn.conv2d(self.inputs, self.weights, [1, 1, 1, 1], padding=DEFAULT_PADDING)
            _pre = tf.nn.bias_add(_conv, self.biases)
            if self.apply_batch_norm:
                _pre = _bn(_pre, _pre.get_shape()[-1].value)
            self.outputs = self.activation(_pre)
            #
            # w_min = tf.reduce_min(self.weights)
            # w_max = tf.reduce_max(self.weights)
            # w_norm = (self.weights - w_min) / (w_max - w_min)
            # w_norm_trans = tf.transpose(w_norm, [3, 0, 1, 2])
            # tf.summary.image('{}/kernel'.format(self.name), w_norm_trans)


class AffineLayer(Layer):
    """
    Fully Connected Layer
    """
    def __init__(self, name, flatten_inputs=False, final_layer=False, weight_decay=None):
        super(AffineLayer, self).__init__(name)
        self.final_layer = final_layer
        self.flatten_inputs = flatten_inputs
        self.weight_decay = weight_decay

    def set_inputs(self, inputs):
        self.inputs = inputs
        if self.flatten_inputs:
            self.flatten()
        self.set_outputs()

    def set_outputs(self):
        with tf.name_scope(self.name):

            input_dim = self.inputs.get_shape()[1].value
            output_dim = 10 if self.final_layer else int(input_dim/2)

            weights = _weights(
                [input_dim, output_dim],
                2. / (input_dim + output_dim) ** 0.5, decay=self.weight_decay)
            biases = tf.Variable(tf.zeros([output_dim]), name='biases')

            if self.final_layer:
                self.outputs = tf.add(tf.matmul(self.inputs, weights), biases)
            else:
                self.outputs = self.activation(tf.matmul(self.inputs, weights) + biases)


class PoolLayer(Layer):
    """
    Layer of non-overlapping 1D pools of inputs.
    SRC: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    Stride, ksize = 2
    """
    def __init__(self, name):
        super(PoolLayer, self).__init__(name)

    def set_outputs(self):
        with tf.name_scope(self.name):
            self.outputs = tf.nn.max_pool(
                self.inputs,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding=DEFAULT_PADDING,
                name=self.name)


# OTHER HELPER METHODS

def _bn(inputs, output_dim):
    """Batch normalization on convolutional layers."""
    beta = tf.Variable(tf.constant(0.0, shape=[output_dim]), name='beta')
    gamma = tf.Variable(tf.constant(1.0, shape=[output_dim]), name='gamma')
    batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2], name='moments')
    epsilon = 1e-3
    return tf.nn.batch_normalization(
        inputs, batch_mean, batch_var, beta, gamma, epsilon, 'bn'
    )


def _weights(shape, stddev=0.1, decay=None, name='weights'):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))
    if decay is not None:
        weights = tf.multiply(tf.nn.l2_loss(weights), decay)
    return weights


def _biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.0, shape=shape, name=name))
