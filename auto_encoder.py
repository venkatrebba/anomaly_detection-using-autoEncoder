"""
Definition of Auto Encoder model

Author: Venkat Rebba <rebba498@gmail.com>
"""

import tensorflow as tf
from tensorflow.python.keras.initializers import he_normal
from tensorflow.image import ResizeMethod
from tensorflow.train import AdamOptimizer as Adam


def conv_layer(input_layer, size_in, size_out, name="conv"):
    """
    This function is to define convloution layer with
    following default parmameters
        Filter size:		5*5
        Padding:			SAME
        Inititalization: 	HE Normal
        Activation:			LeakyRelu(Alpha=0.5)
    """
    with tf.name_scope(name):
        # Defining fliter weights
        weights = tf.get_variable(
                        name + "/" + "weights",
                        dtype=tf.float32,
                        shape=[5, 5, size_in, size_out],
                        initializer=he_normal(seed=8)
                        )
        # Defining biases
        biases = tf.get_variable(
                        name + "/" + "biases",
                        dtype=tf.float32,
                        shape=[size_out],
                        initializer=tf.constant_initializer(0.01)
                        )
        # Convolving the input with weights
        conv = tf.nn.conv2d(input_layer, weights,
                            strides=[1, 1, 1, 1], padding="SAME")
        # Applying leakey relu activation function
        act = tf.nn.leaky_relu(conv + biases, alpha=0.5, name="leakyrelu")

        # Adding historgrams to visualize in tensorboard
        tf.summary.histogram("weights", weights)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", act)
        return act


def max_pool_layer(input_layer, filter_size, stride_size, name):
    """
    In this function max pool layer is defined
    """
    max_pool = tf.nn.max_pool(input_layer,
                              ksize=[1, filter_size, filter_size, 1],
                              strides=[1, stride_size, stride_size, 1],
                              padding="SAME", name=name)
    return max_pool


def upsample_layer(input_layer, output_size, name):
    """
    This function does upsampling of given input
    """
    upsample = tf.image.resize_images(input_layer, size=output_size,
                                      method=ResizeMethod.NEAREST_NEIGHBOR)
    return upsample


def get_optimizer_op(logits, labels, learning_rate):
    """
    Function computes loss value and minimize it through optimizer
    """
    with tf.name_scope("loss"):
        loss = tf.reduce_mean(
                    tf.losses.mean_squared_error(labels, logits),
                    name="loss"
                    )
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        train_step = Adam(learning_rate, name='Adam').minimize(
                                     loss,
                                     global_step=tf.train.get_global_step()
                                     )

    return loss, train_step


def auto_encoder_model(features, labels, params, mode):
    """
    Auto Encoder model definition
    """

    # Converting input to float
    features = tf.to_float(features)
    x = tf.reshape(features, [-1, 224, 224, 3], name="x")
    tf.summary.image('input', x)

    y = tf.identity(x)

    # Encoder
    # input size: 224*224*3
    conv1 = conv_layer(x, 3, 32, "conv1")

    # 224*224*32
    pool1 = max_pool_layer(conv1, 2, 2, "pool1")

    # 112*112*32
    conv2 = conv_layer(pool1, 32, 64, "conv2")

    # 112*112*64
    pool2 = max_pool_layer(conv2, 2, 2, "pool2")

    # 56*56*64
    conv3 = conv_layer(pool2, 64, 128, "conv3")

    # 56*56*128
    pool3 = max_pool_layer(conv3, 2, 2, "pool3")

    # Decoder
    # 28*28*128
    upsampl1 = upsample_layer(pool3, (56, 56), "upsample1")

    # 56*56*128
    conv4 = conv_layer(upsampl1, 128, 64, "conv4")

    # 56*56*64
    upsampl2 = upsample_layer(conv4, (112, 112), "upsample2")

    # 112*112*64
    conv5 = conv_layer(upsampl2, 64, 32, "conv5")

    # 112*112*32
    upsampl3 = upsample_layer(conv5, (224, 224), "upsample3")

    # 224*224*32
    logits = conv_layer(upsampl3, 32, 3, "logits")
    tf.summary.image('logit_image', logits, 3)

    learning_rate = params['learning_rate']
    loss, train_step = get_optimizer_op(logits, y, learning_rate)

    # Predictions, predict the loss of input image
    predictions = {
        "loss": loss,
        "logits": logits
    }

    # when mode == 'infer'
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            'frame_pred': tf.estimator.export.PredictOutput(predictions)
            }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs
            )

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops={
                'eval_loss': tf.metrics.mean_squared_error(y, logits)
                }
        )

    # Logging hook
    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)

    # default train mode
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_step,
        training_hooks=[logging_hook]
        )
