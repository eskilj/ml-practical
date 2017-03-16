import tensorflow as tf
import os
import datetime
import layer
from model import Model
import input
import numpy as np

# check necessary environment variables are defined
assert 'MLP_DATA_DIR' in os.environ, (
    'An environment variable MLP_DATA_DIR must be set to the path containing'
    ' MLP data before running script.')
assert 'OUTPUT_DIR' in os.environ, (
    'An environment variable OUTPUT_DIR must be set to the path to write'
    ' output to before running script.')

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', 'tf-log',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def graph_summary(error, accuracy, name, graph):
    tf.summary.scalar('error', error)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    # create objects for writing summaries and checkpoints during training
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(os.environ['OUTPUT_DIR'], timestamp)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'train'), graph=graph)
    valid_writer = tf.summary.FileWriter(os.path.join('tf-log', name, 'valid'), graph=graph)
    saver = tf.train.Saver()
    return summary_op, train_writer, valid_writer, saver, checkpoint_dir, exp_dir


def train_graph(model):
    graph = tf.Graph()
    with graph.as_default():

        train_data, valid_data = input.inputs()
        inputs, targets = input.placeholder()

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

        summary_op, train_writer, valid_writer, saver, checkpoint_dir, exp_dir = layer.graph_summary(error, accuracy, model.name, graph)

    sess = tf.Session(graph=graph)

    sess.run(tf.global_variables_initializer())
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

        if (b % 400 == 0) or (b == last_batch):
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
    model = Model.trial()
    if tf.gfile.Exists(os.path.join(FLAGS.train_dir, model.name)):
        print('Deleting previous summary directory.')
        tf.gfile.DeleteRecursively(os.path.join(FLAGS.train_dir, model.name))
    tf.gfile.MakeDirs(os.path.join(FLAGS.train_dir, model.name))
    train_graph(model)


if __name__ == '__main__':
    tf.app.run()
