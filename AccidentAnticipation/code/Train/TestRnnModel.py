# Author : Ketan Kokane <kk7471@rit.edu>
from sklearn.metrics import classification_report

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
def _test(video_sequence_tensor, rnn):
    """
    This function scope is over a video clip,
    its supposed to get a video frame tensor generated using Genaret_feature
    :param true_value_tensor:
    :param video_sequence_tensor: [n, 4096], where n is the length of video
    sequence
    :return:
    """
    hidden = rnn.initHidden()
    prediction_list = []
    for i in range(video_sequence_tensor.size()[0]):
        # for ith frame in the video frame
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
        prediction_list.append(prediction_tensor)
        # get prediction for every frame
    prediction_tensor = torch.cat(prediction_list)
    # get prediction for every frame

    prediction_tensor.data.numpy()

    return prediction_tensor  # return  total loss for the current video sequence


def test():
    rnn = RNN(4096, n_hidden)
    rnn.load_state_dict(torch.load(
            '../../trained_models/rnn_optimized40.model'))
    rnn.eval()

    # load the dataset,
    dataset = get_dataset_path('trainair')
    video_clip_target = load_dataset(dataset)
    # create the model

    for data_item in video_clip_target:
        feature_tensors = data_item[0]
        targets = data_item[1]

        output = _test(feature_tensors, rnn)
        output = output.data.numpy()
        output[output >= 0.50] = 1
        output[output < 0.50] = 0
        output = output.astype(int)

        print('current clip accuracy = ', np.mean(output == targets))
        print(classification_report(targets, output))




def get_video_clip_from_test_set(video_clip_path):
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


def get_dataset_path(test_train):
    if test_train == 'test':
        #  TODO: Need to change this to load the actual dataset
        lst = []
        for _ in range(1, 30):
            video_path = f'../../dataset/test/videoclips/clip_' \
                         f'{_}/feature_tensors.txt'
            target_path = f'../../dataset/test/groundtruth/clip_{_}.txt'
            lst.append((video_path, target_path))
        return lst
    else:
        #  TODO: Need to change this to load the actual dataset
        lst = []
        for _ in range(1, 30):
            video_path = f'../../dataset/train/videoclips/clip_' \
                         f'{_}/feature_tensors.txt'
            target_path = f'../../dataset/train/groundtruth/clip_{_}.txt'
            lst.append((video_path, target_path))
        return lst


def load_dataset(dataset_paths):
    #  TODO: Need to change this to load the actual dataset
    lst = []
    print('generating features for all video clips')
    for dataset_path in dataset_paths:
        video_clip = np.loadtxt(dataset_path[0], dtype=np.float32)
        targets = get_targets_tensor(dataset_path[1])
        # TODO: make sure feature_tensors.device == cuda
        feature_tensors = torch.from_numpy(video_clip)
        lst.append((feature_tensors, targets))
    print('done generating features for all video clips', len(lst))
    return lst

def get_targets_tensor(file_path):
    targets = np.loadtxt(file_path, dtype = np.float32)
    targets = targets.astype(int)
    return targets



if __name__ == '__main__':
    # when this script is executed, it is supposed to generate a model file
    # using the train portion of the dataset
    test()
