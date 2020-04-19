# Author : Ketan Kokane <kk7471@rit.edu>

import torch
from PIL import Image
import  os
from Feature_Extraction.VGG_16 import MODEL_PATH, preprocess, getModel, VGG_16

import numpy as np

def get_features_tensors(image):
    """

    :param image: image read using PIL, has to be 3 * H * W
    :return: return torch tensor features created using VGG model
    [1 * 4096]
    """
    vgg = getModel('../' + MODEL_PATH)
    input_tensor = preprocess(image)
    # return [1 * 4096]
    # unsqueeze turns it into [1, 3, 244, 244] which can be thought of batch
    # size of 1
    input_batch = input_tensor.unsqueeze(0)
    features = vgg(input_batch)
    # return 1 * 4096 dims feature tensor
    return features


def get_features_tensors_for_video_clip(video_clip):
    """

    :param image: image read using PIL, has to be 3 * H * W
    :return: return torch tensor features created using VGG model
    op [10, 4096] for an video clip of length 10, meaning 10 frames
    """
    vgg = getModel('../' + MODEL_PATH)
    arr = []
    for image in video_clip:
        # the op of the preprocess(image) is a tensor of [3, 244, 244]
        # use of unsqueeze changes it to [1, 3, 244, 244]
        input_tensor = preprocess(image).unsqueeze(0)
        arr.append(input_tensor)
    input_batch = torch.cat(arr)
    features = vgg(input_batch)
    return features


def get_dataset_path():
    #  TODO: Need to change this to load the actual dataset
    lst = []
    for _ in range(1, 97):
        video_path = f'../../dataset/train/videoclips/clip_' \
                     f'{_}/'
        target_path = f'../../dataset/train/groundtruth/clip_{_}.txt'
        lst.append((video_path, target_path))
    return lst

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


def load_dataset(dataset_paths):
    lst = []
    print('generating features for all video clips')
    for dataset_path in dataset_paths:
        video_clip = get_video_clip_from_training_set(dataset_path[0])
        # print(dataset_path[0])
        # TODO: make sure feature_tensors.device == cuda
        feature_tensors = get_features_tensors_for_video_clip(video_clip)
        np.savetxt(dataset_path[0] + 'feature_tensors.txt',\
                                    feature_tensors.data.numpy())
    print('done generating features for all video clips')
    return lst




if __name__ == '__main__':
    load_dataset(get_dataset_path())





