"""
Train & eval code of AutoEncoder model for Video Anomlay detection

"""

import os
import time
import datetime
from os.path import abspath
import tensorflow as tf
from argparse import ArgumentParser
from auto_encoder import auto_encoder_model
from ae_ai_input import parse_fn, input_fn, serving_inp_rcv

# Command Line argument parser.
parser = ArgumentParser(description='AutoEncoder Model Train / Evaluate')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# List of required CL arguments.
# The directory containing the training images.
required_args.add_argument("-d", "--traindir",
                           help="Directory of the image files to train",
                           required=True)

required_args.add_argument("-e", "--evaldir",
                           help="Directory of the image files to evaluate",
                           required=False)

required_args.add_argument("-m", "--mean",
                           help="Mean of training dataset",
                           required=True)

required_args.add_argument("-s", "--std",
                           help="Standard deviation of training dataset",
                           required=True)

required_args.add_argument("-b", "--batch",
                           help="Batch size to train the model",
                           required=True)

required_args.add_argument("-i", "--epochs",
                           help="Number of epochs to be trained",
                           required=True)

args = parser.parse_args()

MEAN = float(args.mean)
STD = float(args.std)

# The directory where the trained model will be saved.
auto_encoder_dir = str(os.getcwd() + os.sep +
                       "auto_encoder/{}".format(time.time()))

# DEBUG
print("auto_encoder_dir = ", auto_encoder_dir)

# Training dataset deirectory
train_dir = args.traindir
train_dir = abspath(train_dir)

# The eval directory
eval_dir = args.evaldir

# Loss tensor for logging
tensors_to_log = {"loss": "loss/loss:0"}

# Log for TensorBoard.
logging_hook = tf.train.LoggingTensorHook(
       tensors=tensors_to_log, every_n_iter=1)

# Logging
tf.logging.set_verbosity(tf.logging.INFO)

# batch size for trianing
batch = int(args.batch)

# Larning rate for model
learning_rate = 0.001

# No of epochs to be trained
epochs = int(args.epochs)

# num of cores in the system, it is used for parallel mapping in input function
num_parallel_calls = 12

# Parameters to be passed on to the input function.
input_params = {
        'train_dir': train_dir,
        'eval_dir': eval_dir,
        'batch': batch,
        'mean': MEAN,
        'std': STD,
        'learning_rate': learning_rate,
        'num_parallel_calls': num_parallel_calls
        }

# Instantiate an estimator object.
ae_model = tf.estimator.Estimator(
         model_fn=auto_encoder_model,
         model_dir=auto_encoder_dir,
         params=input_params
         )

start = time.time()

for epoch in range(0, epochs):
    # Training the model
    ae_model.train(
        input_fn=lambda mode: input_fn(mode=mode, input_params=input_params),
        hooks=[logging_hook]
        )

    # Logs to know training time
    print("*"*50, "\nTime for epoch {} : {} sec".
          format(epoch+1, time.time() - start),
          "\ncurrent time: {} \n".format(datetime.datetime.now()),
          "*"*50
          )

    # Validating the model
    print('Evaluaiton of model')
    ae_model.evaluate(
        input_fn=lambda mode: input_fn(mode=mode, input_params=input_params),
        )

    start = time.time()

    # Saving inference model for every 5 epochs
    if(epoch % 5 == 0):
        # Export model for inference
        _exp = ae_model.export_savedmodel(
                "ae_prd_mdl_" + str(batch),
                serving_input_receiver_fn=serving_inp_rcv
                )

# Export the model for inference
_exp = ae_model.export_savedmodel(
            "ae_prd_mdl_" + str(batch),
            serving_input_receiver_fn=serving_inp_rcv
            )
