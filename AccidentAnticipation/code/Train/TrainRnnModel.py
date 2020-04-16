# Author : Ketan Kokane <kk7471@rit.edu>


import os

import torch
import torch.nn as nn
from PIL import Image

from Feature_Extraction.VGG_16 import MODEL_PATH


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, 1)
        self.sigmoid = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 128

rnn = RNN(4096, 128)

learning_rate = 0.005


# If set too high, it might explode. If too low, it might not learn

def _train(true_value_tensor, video_sequence_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(video_sequence_tensor.size()[0]):
        # for every frame in video sequence
        prediction_tensor, hidden = rnn(video_sequence_tensor[i], hidden)
    # get prediction for every frame

    loss = criterion(prediction_tensor, true_value_tensor)
    # we want Exponential Loss here
    loss.backward()  # backpropogate

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return prediction_tensor, loss.item()  # return final predicted and total loss


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


def train():
    vgg_model = torch.load('../' + MODEL_PATH)
    hidden_state = torch.zeros(128, dtype=torch.int32)

    for video_clip_path in [1,2,3]:
        frames = get_video_clip_from_training_set(video_clip_path)



