import os
import random

import numpy as np
import torch
from PIL import Image

from Feature_Extraction.VGG_16 import preprocess, getModel, MODEL_PATH


def get_random_video_clip():
    """
    This function is supposed to read in from the during training,
    a random video clip from train folder of the dataset

    returns: a list of frames inside the video clip
    """
    video_clip_path = '../../dataset/train/videoclips/clip_1/'
    frames = []
    for filename in os.listdir(video_clip_path):
        # put some condition so only .jfg files are read
        image = Image.open(video_clip_path + filename)
        frames.append(image)
    return frames


def get_features_from_vgg_16(image_frame, model):
    """

    :param image_frame:
    :param model:
    :return:
    """
    image_tensor = preprocess(image_frame)
    image_tensor_batch = image_tensor.unsqueeze(0)
    feature_tensor = model(image_tensor_batch)
    return feature_tensor


def rnn_model(features, hidden_state, train=False):
    output_proba = random.random()
    return output_proba, np.random.rand(128)


def predict():
    """

    :return:
    """

    frames = get_random_video_clip()
    model = getModel('../' + MODEL_PATH)
    hidden_state = torch.zeros(128, dtype=torch.int32)

    # plt.ion()
    for image_frame in frames:
        feature_tensor = get_features_from_vgg_16(image_frame, model)
        accident_proba, hidden_state = rnn_model(feature_tensor, hidden_state,
                                                 train=False)
        # plt.close()
        # plt.imshow(image_frame)
        if accident_proba > 0.90:
            print('Accident Happened')
            break
        # plt.show()


if __name__ == '__main__':
    predict()
