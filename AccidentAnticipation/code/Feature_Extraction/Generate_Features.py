# Author : Ketan Kokane <kk7471@rit.edu>

import torch
from PIL import Image

from Feature_Extraction.VGG_16 import MODEL_PATH, preprocess, getModel, VGG_16


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


if __name__ == '__main__':
    img_path = '../../dataset/train/videoclips/clip_1/000017.jpg'
    input_image = Image.open(img_path)
    print(get_features_tensors(input_image).shape)
    frames = [input_image, input_image]
    print(get_features_tensors_for_video_clip(frames).shape)
