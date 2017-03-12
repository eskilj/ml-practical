import tensorflow as tf

DEFAULT_PADDING = 'SAME'


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=DEFAULT_PADDING)


def max_pool(x):
    """Layer outputting the maximum of non-overlapping 1D pools of inputs."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding=DEFAULT_PADDING)


def relu(x, name='relu'):
    """Layer implementing an element-wise rectified linear transformation."""
    return tf.nn.relu(x, name=name)
