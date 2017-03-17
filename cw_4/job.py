#!/disk/scratch/mlp/miniconda2/bin/python
import tensorflow as tf
import os
import numpy as np
import mlp.data_providers as data_providers

def_padding = 'SAME'
num_thr = 8


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tf-log',
                           """Directory where to write event logs """
                           """and checkpoint.""")

tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")

# DATA CONSTANTS
img_size = 32
num_cls = 10
num_ch = 3


def _inputs():
    """
    Load data from Data Provider

    Returns:
      train_data: Training Data
      valid_data: Validation Data
    """

    train_data = data_providers.CIFAR10DataProvider('train', batch_size=FLAGS.batch_size)
    valid_data = data_providers.CIFAR10DataProvider('valid', batch_size=FLAGS.batch_size)
    return train_data, valid_data


def placeholder():
    inp = tf.placeholder(
        tf.float32,
        shape=[None, img_size, img_size, num_ch])
    targ = tf.placeholder(tf.float32, shape=[None, num_cls])
    return inp, targ


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
    def __init__(self, weights, biases, name, apply_batch_norm=False):
        super(Conv2dLayer, self).__init__(name)
        self.weights = weights
        self.biases = biases
        self.apply_batch_norm = apply_batch_norm

    def set_outputs(self):
        with tf.name_scope(self.name):

            self.biases = _biases(self.biases)
            self.weights = _weights(self.weights)

            _conv = tf.nn.conv2d(self.inputs, self.weights, [1, 1, 1, 1], padding=def_padding)
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
    def __init__(self, name, flatten_inputs=False, final_layer=False):
        super(AffineLayer, self).__init__(name)
        self.final_layer = final_layer
        self.flatten_inputs = flatten_inputs
        self.weights = None

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
                2. / (input_dim + output_dim) ** 0.5)
            self.weights = weights
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
                padding=def_padding,
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


def _weights(shape, stddev=0.1, name='weights'):
    weights = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev, name=name))
    return weights


def _biases(shape, name='biases'):
    return tf.Variable(tf.constant(0.0, shape=shape, name=name))


class Model:

    def __init__(self, name, layers, activation=tf.nn.relu, train_epochs=10, initial_lr=0.001, l2_loss=None):
        self.name = name or 'unnamed model'
        self.layers = layers
        self.train_epochs = train_epochs
        self.initial_lr = initial_lr
        self.activation = activation
        self.l2_loss = l2_loss

    def get_layers(self, inputs):
        for i, layer in enumerate(self.layers):
            layer.set_activation(self.activation)
            if i == 0:
                layer.set_inputs(inputs)
            else:
                layer.set_inputs(self.layers[i - 1].outputs)

        return self.layers[-1].outputs

    def get_l2_losses(self):
        weights = []
        for layer in self.layers:
            if isinstance(layer, (Conv2dLayer, AffineLayer)):
                weights.append(tf.nn.l2_loss(layer.weights))
        return self.l2_loss * sum(weights)

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


def graph_summary(error, accuracy, name, graph):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    # create objects for writing summaries and checkpoints during training
    exp_dir = os.path.join('tf-log', name)
    checkpoint_dir = exp_dir
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'train'), graph=graph)
    valid_writer = tf.summary.FileWriter(os.path.join(exp_dir, 'valid'), graph=graph)
    saver = tf.train.Saver()
    return summary_op, train_writer, valid_writer, saver, checkpoint_dir, exp_dir


def train_graph(model):
    graph = tf.Graph()
    with graph.as_default():

        train_data, valid_data = _inputs()
        inputs, targets = placeholder()

        _valid_inputs = valid_data.inputs.reshape((10000, -1, 3), order='F')
        _valid_inputs = _valid_inputs.reshape((10000, 32, 32, 3))
        valid_targets = valid_data.to_one_of_k(valid_data.targets)

        tf.summary.image('img', inputs)
        outputs = model.get_layers(inputs)

        with tf.name_scope('error'):
            error = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(outputs, targets))

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
                        tf.float32))

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(model.initial_lr).minimize(error)

        summary_op, train_writer, valid_writer, saver, checkpoint_dir, exp_dir = graph_summary(error, accuracy, model.name, graph)
        init = tf.global_variables_initializer()

    sess = tf.Session(graph=graph, config=tf.ConfigProto(intra_op_parallelism_threads=num_thr))

    sess.run(init)
    last_batch = (model.train_epochs * 800) - 1

    train_accuracy = np.zeros(model.train_epochs)
    train_error = np.zeros(model.train_epochs)
    valid_accuracy = np.zeros(model.train_epochs)
    valid_error = np.zeros(model.train_epochs)
    step = 0

    for e in range(model.train_epochs):
        for b, (input_batch, target_batch) in enumerate(train_data):

            input_batch = input_batch.reshape((50, -1, 3), order='F')
            input_batch = input_batch.reshape(50, 32, 32, 3)

            # do train step with current batch
            _, summary, batch_error, batch_acc = sess.run(
                [train_step, summary_op, error, accuracy],
                feed_dict={inputs: input_batch, targets: target_batch})

            # add symmary and accumulate stats
            train_writer.add_summary(summary, step)
            train_error[e] += batch_error
            train_accuracy[e] += batch_acc
            step += 1
        # normalise running means by number of batches
        train_error[e] /= train_data.num_batches
        train_accuracy[e] /= train_data.num_batches

        if (step % 100 == 0) or (step == last_batch):
            # evaluate validation set performance
            valid_summary, valid_error[e], valid_accuracy[e] = sess.run(
                [summary_op, error, accuracy],
                feed_dict={inputs: _valid_inputs, targets: valid_targets})
            valid_writer.add_summary(valid_summary, step)
            # checkpoint model variables
            saver.save(sess, os.path.join(checkpoint_dir, 'model.ckpt'), step)
            # write stats summary to stdout
            print('Epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
                  .format(e + 1, train_error[e], train_accuracy[e]))
            print('          err(valid)={0:.2f} acc(valid)={1:.2f}'
                  .format(valid_error[e], valid_accuracy[e]))
    # close writer and session objects
    train_writer.close()
    valid_writer.close()
    sess.close()

    # save run stats to a .npz file
    np.savez_compressed(
        os.path.join(exp_dir, 'run.npz'),
        train_error=train_error,
        train_accuracy=train_accuracy,
        valid_error=valid_error,
        valid_accuracy=valid_accuracy
    )


def main(argv=None):

    lrs = [0.005]
    acs = [tf.nn.relu]
    f_sizes = [3]
    num_f = [24]
    epochs = 40
    wd = 0.005

    for lr in lrs:
        for ac in acs:
            for fs in f_sizes:
                for nf in num_f:
                    layers = [
                        Conv2dLayer([fs, fs, 3, nf], [nf], 'conv_1', True),
                        PoolLayer('pool_1'),
                        Conv2dLayer([fs, fs, nf, nf], [nf], 'conv_2', True),
                        PoolLayer('pool_2'),
                        AffineLayer(name='fc_1', flatten_inputs=True),
                        AffineLayer(name='fc_2'),
                        AffineLayer(name='output', final_layer=True)
                    ]

                    _mo = Model(
                        'stage2/{},fs={},nf={},lr={},wd={}'.format(ac.func_name, fs, nf, lr, wd),
                        layers=layers,
                        activation=ac,
                        train_epochs=epochs,
                        initial_lr=lr,
                        l2_loss=wd
                    )

                    if tf.gfile.Exists(os.path.join(FLAGS.train_dir, _mo.name)):
                        print('Deleting previous summary directory.')
                        tf.gfile.DeleteRecursively(
                            os.path.join(FLAGS.train_dir, _mo.name))
                    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, _mo.name))

                    train_graph(_mo)


if __name__ == '__main__':
    tf.app.run()
