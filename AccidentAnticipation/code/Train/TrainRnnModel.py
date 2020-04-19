# Author : Ketan Kokane <kk7471@rit.edu>


import math
import os
import random
import time

import numpy as np
import torch.optim as optim

import torch.nn as nn
from Feature_Extraction.VGG_16 import *
from Feature_Extraction.Generate_Features import *
from Train.RNN import *

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_hidden = 128
epochs = 5


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
    prediction_list = []
    for i in range(video_sequence_tensor.size()[0]):
        # for ith frame in the video frame
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
        prediction_list.append(prediction_tensor)
    # get prediction for every frame
    prediction_tensor = torch.cat(prediction_list)
    # true_value_tensor = true_value_tensor[:2]
    loss = criterion(prediction_tensor, true_value_tensor)
    # we want Exponential Loss here
    loss.backward()  # backpropogate

    optimizer.step()

    return loss.item()  # return  total loss for the current video sequence


def load_dataset(dataset_paths):
    lst = []
    print('generating features for all video clips')
    for dataset_path in dataset_paths:
        video_clip = get_video_clip_from_training_set(dataset_path[0])
        targets = get_targets_tensor(dataset_path[1])
        # TODO: make sure feature_tensors.device == cuda
        feature_tensors = get_features_tensors_for_video_clip(video_clip)
        lst.append((feature_tensors, targets))
    print('done generating features for all video clips', len(lst))
    return lst


def train():
    # load the dataset,
    dataset = get_dataset_path()
    # create the model
    video_clip_target = load_dataset(dataset)
    rnn = RNN(4096, n_hidden)

    # TODO: Change this later to Exponential Loss
    criterion = nn.BCELoss()

    optimizer = optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()


    for epoch in range(1, epochs + 1):
        random.shuffle(video_clip_target)  # random the video clips (so the model does not
        # memorize anything

        for data_item in video_clip_target:
            feature_tensors = data_item[0]
            targets = data_item[1]

            loss = _train(feature_tensors, targets, rnn , criterion, optimizer)

            current_loss += loss

        all_losses.append(current_loss)
        print('epochs=', epoch, 'total Loss in this epoch=', current_loss,
              'time since start=', timeSince(start))
        current_loss = 0
        # Save the model
        torch.save(rnn.state_dict(), '../../trained_models/rnn_optimized' + str(
                epoch) +
                   '.model')

    # print(current_loss)# put this as an np array and store it in a file
    plt.figure()
    plt.plot(all_losses)
    plt.show()
    plt.savefig('total_loss.png')


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
        if filename.endswith('.jpg'):
            image = Image.open(video_clip_path + filename)
            frames.append(image)
    return frames


def get_dataset_path():
    #  TODO: Need to change this to load the actual dataset
    lst = [('../../dataset/train/videoclips/clip_1/',
            '../../dataset/train/groundtruth/clip_1.txt'),
           ('../../dataset/train/videoclips/clip_2/',
            '../../dataset/train/groundtruth/clip_2.txt')
           ,('../../dataset/train/videoclips/clip_3/',
            '../../dataset/train/groundtruth/clip_3.txt')
        , ('../../dataset/train/videoclips/clip_4/',
           '../../dataset/train/groundtruth/clip_4.txt')
        , ('../../dataset/train/videoclips/clip_5/',
           '../../dataset/train/groundtruth/clip_5.txt'),
           ('../../dataset/train/videoclips/clip_6/',
            '../../dataset/train/groundtruth/clip_6.txt')
        , ('../../dataset/train/videoclips/clip_7/',
           '../../dataset/train/groundtruth/clip_7.txt')
        , ('../../dataset/train/videoclips/clip_8/',
           '../../dataset/train/groundtruth/clip_8.txt')
        , ('../../dataset/train/videoclips/clip_9/',
           '../../dataset/train/groundtruth/clip_9.txt')
        , ('../../dataset/train/videoclips/clip_10/',
           '../../dataset/train/groundtruth/clip_10.txt')
           ]
    return lst


def get_targets_tensor(file_path):
    return torch.from_numpy(np.loadtxt(file_path, dtype = np.float32) )


if __name__ == '__main__':
    # when this script is executed, it is supposed to generate a model file
    # using the train portion of the dataset
    train()
