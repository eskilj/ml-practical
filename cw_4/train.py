import tensorflow as tf
import os
import layer
from model import Model
import input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tf-log',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def train_graph(model):
    graph = tf.Graph()
    with graph.as_default():

        train_data, valid_data = input.inputs()
        inputs, targets = input.placeholder()

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

        summary_op, train_writer, valid_writer = layer.graph_summary(error, accuracy, model.name, graph)
        init = tf.global_variables_initializer()

    sess = tf.InteractiveSession(graph=graph)

    _valid_inputs = valid_data.inputs.reshape((10000, -1, 3), order='F')
    _valid_inputs = _valid_inputs.reshape((10000, 32, 32, 3))
    valid_targets = valid_data.to_one_of_k(valid_data.targets)

    sess.run(init)
    for epoch in range(model.train_epochs):
        print('Epoch {} / {}'.format(epoch+1, model.train_epochs))
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
    model = Model.trial()
    if tf.gfile.Exists(os.path.join(FLAGS.train_dir, model.name)):
        print('Deleting previous summary directory.')
        tf.gfile.DeleteRecursively(os.path.join(FLAGS.train_dir, model.name))
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, model.name))
    train_graph(model)


if __name__ == '__main__':
    tf.app.run()
