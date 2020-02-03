"""
The input pipeline and utility functions of Auto Encoder model

Author: Rebba Venkatarao <rebba498@gmail.com>
"""

import tensorflow as tf
import os


def parse_fn(image_path):
    """
    This function is used to convert the image filenames to pixel format
    and apply data pre-processing stepss
    """
    image_string = tf.read_file(image_path)
    image = tf.image.decode_jpeg(
                image_string,
                channels=3,
                dct_method='INTEGER_ACCURATE')
    image = tf.image.convert_image_dtype(image, tf.uint8)
    image = tf.image.resize_images(image, [224, 224])
    return image


def input_fn(mode, input_params):
    """
    This function returns the correct iterator object for 'train' or
    'eval' based on the mode with which this is called.
    """

    input_files = []  # List of train filenames.
    num_parallel_calls = input_params['num_parallel_calls']
    input_dir = input_params['train_dir']
    batch = input_params['batch']
    MEAN = input_params['mean']
    STD = input_params['std']
    input_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir) if f.endswith('.jpg')]

    dataset = tf.data.Dataset.from_tensor_slices(
                tf.constant(input_files)
                )

    dataset = dataset.shuffle(1024)
    dataset = dataset.map(parse_fn, num_parallel_calls=num_parallel_calls)
    # Normalizing dataset
    dataset = dataset.map(lambda image: (image-MEAN)/STD,
                          num_parallel_calls=num_parallel_calls
                          )
    dataset = dataset.batch(batch_size=batch)
    # Fetch next 2 batches before GPU request
    dataset = dataset.prefetch(2)

    # creating one shot iterator
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()
    return images, None


def serving_inp_rcv():
    """
    Function to cereate tensor serving api
    """
    cap_frame = tf.placeholder(dtype=tf.uint8, shape=[224, 224, 3])

    return tf.estimator.export.TensorServingInputReceiver(
            features=cap_frame,
            receiver_tensors=cap_frame
            )
