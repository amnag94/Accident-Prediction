# Author : Ketan Kokane <kk7471@rit.edu>


import os
import random
import time
import numpy as np
import math
import torch.optim as optim
import torch.nn as nn
from Feature_Extraction.VGG_16 import *
from Feature_Extraction.Generate_Features import *
from Train.RNN import *

n_hidden = 128
epochs = 3


# TODO: add Exponential loss, and move it to CUDA
def _train(video_sequence_tensor, true_value_tensor, rnn, criterion, optimizer):
    """
    This function scope is over a video clip,
    its supposed to get a video frame tensor generated using Genaret_feature
    :param true_value_tensor:
    :param video_sequence_tensor: [n, 4096], where n is the length of video
    sequence
    :return:
    """
    hidden = rnn.initHidden()
    # rnn.zero_grad()
    optimizer.zero_grad()

    for i in range(video_sequence_tensor.size()[0]):
        # for ith frame in the video frame
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
    # get prediction for every frame

    loss = criterion(prediction_tensor, true_value_tensor)
    # we want Exponential Loss here
    loss.backward()  # backpropogate

    optimizer.step()

    return loss.item()  # return  total loss for the current video sequence


def train():
    # load the dataset,
    dataset = load_dataset()
    # create the model

    rnn = RNN(4096, n_hidden)

    # TODO: Change this later to Exponential Loss
    criterion = nn.NLLLoss()

    optimizer = optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        random.shuffle(dataset)  # random the video clips (so the model does not
        # memorize anything

        for data_item in dataset:
            video_clip = get_video_clip_from_training_set(data_item[0])
            targets = get_targets(data_item[1])

            feature_tensors = get_features_tensors_for_video_clip(video_clip)

            loss = _train(feature_tensors, targets, rnn , criterion, optimizer)

            current_loss += loss

        all_losses.append(current_loss)
        print('epochs', epochs, 'total NLL Loss in this epoch', current_loss,
              'time since start', timeSince(start))
        current_loss = 0
        # Save the model

    print(current_loss)# put this as an np array and store it in a file


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def get_video_clip_from_training_set(video_clip_path):
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
    train()
