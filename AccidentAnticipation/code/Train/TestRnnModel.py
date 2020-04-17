# Author : Ketan Kokane <kk7471@rit.edu>


import os
import random

import numpy as np

# from Feature_Extraction.VGG_16 import *
from Feature_Extraction.Generate_Features import *
from Train.RNN import *

n_hidden = 128
learning_rate = 0.005
epochs = 3


# TODO: add Exponential loss, and move it to CUDA
def _test(video_sequence_tensor, true_value_tensor, rnn):
    """
    This function scope is over a video clip,
    its supposed to get a video frame tensor generated using Genaret_feature
    :param true_value_tensor:
    :param video_sequence_tensor: [n, 4096], where n is the length of video
    sequence
    :return:
    """
    hidden = rnn.initHidden()
    rnn.zero_grad()

    for i in range(video_sequence_tensor.size()[0]):
        # for ith frame in the video frame
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
    # get prediction for every frame

    loss = criterion(prediction_tensor, true_value_tensor)
    # we want Exponential Loss here
    loss.backward()  # backpropogate

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return loss.item()  # return  total loss for the current video sequence


def test():
    # load the dataset,
    dataset = load_dataset()
    # create the model

    rnn = RNN(4096, n_hidden)

    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)  # random the video clips (so the model does not
        # memorize anything

        for data_item in dataset:
            video_clip = get_video_clip_from_test_set(data_item[0])
            targets = get_targets(data_item[1])

            feature_tensors = get_features_tensors_for_video_clip(video_clip)

            loss = _test(feature_tensors, targets, rnn)

            # print(data_item[0], data_item[1])

            # manage some print statement
    # print some stats after every epoch


def get_video_clip_from_test_set(video_clip_path):
    """
    This function is supposed to read in from the during training,
    a random video clip from train folder of the dataset

    returns: a list of frames inside the video clip
    """
    frames = []
    for filename in os.listdir(video_clip_path):
        # put some condition so only .jfg files are read
        image = Image.open(video_clip_path + filename)
        frames.append(image)
    return frames


def load_dataset():
    lst = [('clip1', 'clip1targets.txt'),
           ('clip2', 'clip2targets.txt'),
           ('clip3', 'clip3targets.txt')]
    return lst


def get_targets(file_path):
    return np.array([0, 0, 0, 0, 1, 1, 1])


if __name__ == '__main__':
    # when this script is executed, it is supposed to generate a model file
    # using the train portion of the dataset
    test()
