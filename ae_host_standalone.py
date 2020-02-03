"""
Integration script of Auto encoder model

Author: Venkat Rebba <rebba498@gmail.com>
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

required_args.add_argument("-f", "--file_name",
                           help="Video file name for  captured video",
                           required=True)

required_args.add_argument("-t", "--timeout",
                           help="Time out in sec",
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

# Get current working directory.
cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))

# Video Codec.
video_codec = cv2.VideoWriter_fourcc(*'XVID')

# AI Model Directory.
ai_model = args.vad_model
MEAN = float(args.mean)
STD = float(args.std)

# Shared Queue Size
queue_size = 20000

# Timeout (secs)
timeout = float(args.timeout)

# Saved Video
saved_frame_file_name = str(args.file_name) + "_" + str(time.strftime(
         "%d_%m_%Y_%H_%M_%S_")) + ".avi"

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


def predict_process(queuein, frame_detect_list, saved_frame_file_name):
    """
    Desc:
    This function starts a parallel process for frame predictions
    The predicted label and frame_count is written on the image and the
    predicted video is saved for easy visualization.
    Args:
    queuein                    : Shared queue between processes for retrieving
                                the captured frames.
    frame_detect_list          : Shared list between processes for storing the
                                predicted labels of each frame.
    saved_frame_file_name      : video file name to save the predicted video.

    Returns:
    None

    """

    try:
        # Load the model to memory.
        predictor_model = tf.contrib.predictor.from_saved_model(ai_model)
    except Exception as e:
        print("Exception: %s while loading the AI model: %s" % (e, ai_model))
        exit(1)

    predict_video = saved_frame_file_name.replace(".", "_predict_0.")
    videowriter_predict = get_video_writer(predict_video)

    frame_count = 0
    start = time.time()
    hours = 0
    with tf.Session() as sess:
        while True:
            frame_count, frame = queuein.get(True)
            # Stop process once "STOP" is sent in queue.
            if str(frame) == "STOP":
                time.sleep(3)
                cv2.destroyAllWindows()
                videowriter_predict.release()
                break

            frame_in = processImg(frame)
            loss_value = predictor_model({"input": sess.run(frame_in)})['loss']
            loss_value = float(loss_value)

            # Labeling the frame based on loss value
            label = "Bad" if(loss_value > MAX_THRESHOLD or
                             loss_value < MIN_THRESHOLD) else "Good"
            # Debuging purpose
            print("%s: %s (%.5f)" % (frame_count, label, loss_value))
            frame_detect_list.append(label)
            cv2.putText(frame, str((frame_count, label)), (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # Display predict window.
            cv2.imshow('capture predict', frame)

            # Save the captured frame with predicted label and frame count.
            videowriter_predict.write(frame)
            cv2.waitKey(1)


def capture_video_settings(camera):
    """
    Desc:
        This function is used to create the camera capture with the
        relevant settings.
    Args:
        camera : Camera ID for video capture object creation

    Returns:
        cap : Created  video capture object.
    """
    try:
        # Capture video from a camera.
        cap = cv2.VideoCapture(camera)
    except Exception as err:
        print("Error opening the camera. Quitting", err)
        exit(1)

    # Set the frame width and height.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    print("Frame Width and Heigth: %s" % str((width, height)))

    # Set the FPS.
    cap.set(cv2.CAP_PROP_FPS, fps)
    print("Camera FPS is set to: %s" % str(fps))

    return cap


def get_video_writer(file_name):
    """
    Desc:
        This function is used to create the video writer object with given
        file name.
    Args:
        file_name : File name for video writer object.

    Returns:
        cv2.VideoWriter() : Created  video writer object.
    """
    return cv2.VideoWriter(file_name, video_codec, fps, (width, height))


if __name__ == "__main__":
    # Start timer.
    start = time.time()

    # Create process.
    manager = multiprocessing.Manager()
    queuein = manager.Queue(queue_size)
    frame_detect_list = manager.list()
    frame_detect_batch_list = manager.list()
    n_process = 1

    # Video Settings.
    cap = capture_video_settings(camera)

    time.sleep(1)

    switch = True
    frame_count = 0

    saved_frame_file_name = os.path.join(cur_dir, saved_frame_file_name)

    videofilewriter = get_video_writer(saved_frame_file_name)

    # Start parallel process for frame predictions.
    pool = multiprocessing.Pool((n_process), predict_process,
                                (queuein, frame_detect_list,
                                saved_frame_file_name))
    start_video = time.time()

    while True:
        # Camera check.
        if cap.isOpened() is False:
            print("camera not open")
            break

        if switch is False:
            print("Switch disabled. Breaking")
            break

        if time.time() - start >= timeout:
            switch = False

        if switch is True:
            # Read frames.
            ret, frame = cap.read()
            if ret is False:
                print("Error reading frames. Quitting")
                break
            frame_count += 1

            # Calculate time stamp
            time_stamp = int(time.time() - start_video)

            # Send frame to queue for prediction.
            queuein.put((frame_count, frame))

            # Display Captured frame
            cv2.putText(frame, "Frame count: %s" % frame_count, (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Time: %ss" % time_stamp, (30, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('capture_live', frame)

            # save the captured frame.
            videofilewriter.write(frame)
            cv2.waitKey(1)

    # Send Stop signal to queuein for ending predict process.
    time.sleep(3)

    # Release all windows.
    cap.release()
    videofilewriter.release()
    cv2.destroyAllWindows()

    # Send stop signal to queue to end the process.
    for i in range(n_process):
        queuein.put((frame_count, "STOP"))

    pool.close()
    pool.join()

    print('Total time:' + str(time.time() - start))
    print('Total frame count : ' + str(frame_count))

    # Calculating Anamolies
    count_bad = len([i for i in frame_detect_list if i != 'Good'])
    anamolies = list(set(frame_detect_list))

    # Percentage of bad frames
    percent_bad = (count_bad/frame_count * 100.0)
    print("Percentage of bad frames is {}%".format(percent_bad))

    # Calculate percentage of each anamoly detected
    dict_anamolies = {}
    for i in anamolies:
        dict_anamolies[i] = frame_detect_list.count(i)

    for key in dict_anamolies:
        percent_temp = (dict_anamolies[key]/frame_count * 100.0)
        print("Percentage of {} frames is {}%".format(key, percent_temp))

    print("Exiting")
