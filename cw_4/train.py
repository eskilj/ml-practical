import tensorflow as tf
import os
import layer
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tf-log/conv',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('graph_name', 'conv4',
                           """Name of graph, for TB """)
tf.app.flags.DEFINE_integer('train_epochs', 10,
                            """Number of epochs to train for.""")


def _train(name):
    graph = tf.Graph()
    with graph.as_default():
        train_data, valid_data = input.inputs()
        inputs, targets = input.placeholder()

        tf.summary.image('img', inputs)

        # Model for training
        conv1 = layer.conv2d(inputs, weights=[5, 5, 3, 4], name='conv1')
        pool1 = layer.max_pool(conv1, name='pool1')

        # conv2 = layer.conv2d(pool1, weights=[5, 5, 4, 4], name='conv2')
        # pool2 = layer.max_pool(conv2, name='pool2')
        #
        # conv3 = layer.conv2d(pool2, weights=[5, 5, 4, 4], name='conv3')
        # pool3 = layer.max_pool(conv3, name='pool3')

        flat, next_input = layer.flatten(pool1)
        fc = layer.fully_connected_layer(flat, next_input, next_input/2)
        outputs = layer.fully_connected_layer(fc, next_input/2, 10, True,
                                              'output')

        with tf.name_scope('error'):
            error = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(outputs, targets))

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1)),
                        tf.float32))

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer().minimize(error)

        summary_op, train_writer, valid_writer = layer.graph_summary(error, accuracy,
                                                               name, graph)

        init = tf.global_variables_initializer()

    sess = tf.InteractiveSession(graph=graph)

    _valid_inputs = valid_data.inputs.reshape((10000, -1, 3), order='F')
    _valid_inputs = _valid_inputs.reshape((10000, 32, 32, 3))
    valid_targets = valid_data.to_one_of_k(valid_data.targets)

    sess.run(init)
    for epoch in range(FLAGS.train_epochs):
        print('Epoch {}'.format(epoch))
        for batch, (input_batch, target_batch) in enumerate(train_data):

            input_batch = input_batch.reshape((50, -1, 3), order='F')
            input_batch = input_batch.reshape(50, 32, 32, 3)

            _, summary = sess.run(
                [train_step, summary_op],
                feed_dict={inputs: input_batch, targets: target_batch})
            train_writer.add_summary(summary,
                                     epoch * train_data.num_batches + batch)
            if (batch % 100 == 0) or (batch == 39999):
                valid_summary = sess.run(
                    summary_op,
                    feed_dict={inputs: _valid_inputs, targets: valid_targets})
                valid_writer.add_summary(valid_summary,
                                         epoch * train_data.num_batches + batch)


def main(argv=None):
    if tf.gfile.Exists(os.path.join(FLAGS.train_dir, FLAGS.graph_name)):
        print('Deleting previous summary directory.')
        tf.gfile.DeleteRecursively(os.path.join(FLAGS.train_dir, FLAGS.graph_name))
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, FLAGS.graph_name))
    _train(FLAGS.graph_name)


if __name__ == '__main__':
    tf.app.run()
