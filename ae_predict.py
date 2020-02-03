"""
Integration script of Auto encoder model

Author: Rebba Venkatarao <venkataraox.rebba@intel.com>
"""

import cv2
import time
import os
import sys
from sys import exit
import socket
import select
import shutil
import multiprocessing
from argparse import ArgumentParser
import tensorflow as tf
import numpy as np
import re
from collections import deque


# Command Line argument parser.
parser = ArgumentParser(description='Video Capture and Analysis on Host')

# List of supported CL arguments.
required_args = parser.add_argument_group('Required Arguments')

# AI Model Directory used for prediction.
required_args.add_argument("-m", "--vad_model",
                           help="AI Model Directory used for prediction",
                           required=True)

required_args.add_argument("-i", "--image",
                           help="Image file name to be predicted",
                           required=True)


required_args.add_argument("-mean", "--mean",
                           help="Dataset mean",
                           required=True)

required_args.add_argument("-std", "--std",
                           help="Dataset std",
                           required=True)

args = parser.parse_args()

# Set the camera device to be used.
camera = 0

# Set the frame rate for camera capture.
fps = 5

# Set width and height of camera.This is the recommended resolution for
# capturing video of SUT.
width = 864
height = 480


# AI Model Directory.
ai_model = args.vad_model
MEAN = float(args.mean)
STD = float(args.std)
image_name = args.image


MAX_THRESHOLD = 0.00099
MIN_THRESHOLD = 0.00009


def processImg(frame):
    """
    Function to pre-process the image
    """
    image = tf.image.convert_image_dtype(frame, tf.uint8)
    image = tf.image.resize_images(image, [224, 224])
    image = (image-MEAN)/STD
    return image


if __name__ == "__main__":
    # Start timer.
    start = time.time()

    frame = cv2.imread(image_name)
	
	try:
        # Load the model to memory.
        predictor_model = tf.contrib.predictor.from_saved_model(ai_model)
    except Exception as e:
        print("Exception: %s while loading the AI model: %s" % (e, ai_model))
        exit(1)

    frame_count = 0
    start = time.time()
    hours = 0
    with tf.Session() as sess:           
            frame_in = processImg(frame)
            loss_value = predictor_model({"input": sess.run(frame_in)})['loss']
            loss_value = float(loss_value)

            # Labeling the frame based on loss value
            label = "Bad" if(loss_value > MAX_THRESHOLD or
                             loss_value < MIN_THRESHOLD) else "Good"
            # Debuging purpose
			print("Prediction, Loss value: {} Prediction: {}, ".format(loss_value, label)
           
    print("Exiting")
