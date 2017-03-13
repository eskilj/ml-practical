import tensorflow as tf
from mlp.data_providers import CIFAR10DataProvider

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_CHANNELS = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 50,
                            """Number of images to process in a batch.""")


def inputs():
    """
    Load data from Data Provider

    Returns:
      train_data: Training Data
      valid_data: Validation Data
    """

    train_data = CIFAR10DataProvider('train', batch_size=FLAGS.batch_size)
    valid_data = CIFAR10DataProvider('valid', batch_size=FLAGS.batch_size)
    return train_data, valid_data


def placeholder():
    inp = tf.placeholder(
        tf.float32,
        shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    targ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
    return inp, targ


def __pre_process_single(img):
    """
    Image processing
    :param img: Input image to be distorted
    :return:
    """

    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=63)
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)

    return img


def pre_process(images):
    images = tf.map_fn(lambda image: __pre_process_single(image), images)

    return images
