import tensorflow as tf
from datetime import datetime
import time

import layer
import model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()

        images, targets = input.placeholder()

        logits = model.inference(images)

        loss = model.loss(logits, targets)

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.argmax(logits, 1), tf.argmax(targets, 1)),
                tf.float32))

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.1
        learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                   global_step, 1000, 0.96)

        with tf.name_scope('train'):
            train_step = get_optimizer(optimizer)(learning_rate).minimize(
                error, global_step=global_step)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for e in range(20):
                running_error = 0.
                running_accuracy = 0.

                for input_batch, target_batch in train_data:
                    _, batch_error, batch_acc = sess.run(
                        [train_step, error, accuracy],
                        feed_dict={inputs: input_batch, targets: target_batch})

                    running_error += batch_error
                    running_accuracy += batch_acc

                running_error /= train_data.num_batches
                running_accuracy /= train_data.num_batches
                print(
                'End of epoch {0:02d}: err(train)={1:.2f} acc(train)={2:.2f}'
                .format(e + 1, running_error, running_accuracy))

                if (e + 1) % 5 == 0:
                    valid_error = 0.
                    valid_accuracy = 0.
                    for input_batch, target_batch in valid_data:
                        batch_error, batch_acc = sess.run(
                            [error, accuracy],
                            feed_dict={
                                inputs: input_batch, targets: target_batch
                                })
                        valid_error += batch_error
                        valid_accuracy += batch_acc

                    valid_error /= valid_data.num_batches
                    valid_accuracy /= valid_data.num_batches
                    print(
                    '                 err(valid)={0:.2f} acc(valid)={1:.2f}'
                    .format(valid_error, valid_accuracy))


def _train():
    graph = tf.Graph()
    name = 'graph'
    with graph.as_default():
        train_data, valid_data = input.inputs()
        inputs, targets = input.placeholder()

        _inputs = tf.reshape(inputs, [-1, input.IMAGE_SIZE, input.IMAGE_SIZE, input.NUM_CHANNELS])
        conv1 = layer.conv2d(_inputs, weights=[5, 5, 3, 64], name='conv1')
        pool1 = layer.max_pool(conv1, name='pool1')
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
    num_epoch = 10
    valid_inputs = valid_data.inputs
    valid_targets = valid_data.to_one_of_k(valid_data.targets)
    sess.run(init)
    for e in range(num_epoch):
        print('Epoch {}'.format(e))
        for b, (input_batch, target_batch) in enumerate(train_data):
            _, summary = sess.run(
                [train_step, summary_op],
                feed_dict={inputs: input_batch, targets: target_batch})
            if b % 100 == 0:
                train_writer.add_summary(summary,
                                         e * train_data.num_batches + b)
                valid_summary = sess.run(
                    summary_op,
                    feed_dict={inputs: valid_inputs, targets: valid_targets})
                valid_writer.add_summary(valid_summary,
                                         e * train_data.num_batches + b)


def main(argv=None):
    _train()


if __name__ == '__main__':
    tf.app.run()
