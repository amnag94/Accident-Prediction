import numpy as np
import os
from skimage import io
import random
import matplotlib.pyplot as plt
import time

def perform_transformation(frame):
     return frame

def get_features_from_vgg_16(transformed_frame):
    return transformed_frame

def rnn_model(features, hidden_state, train = False):
    output_proba = random.random()
    return output_proba, np.random.rand(128)

video_clip_path = 'dataset/train/videoclips/clip_1/'


frames = []
for filename in os.listdir(video_clip_path):
    image = io.imread(video_clip_path + filename)
    frames.append(image)

hidden_state = np.zeros(128)
plt.ion()
plt.show()

# f, ax1 = plt.subplots(1, 1)


for frame in frames:

    print(frame.shape)
    transformed_frame = perform_transformation(frame)
    features = get_features_from_vgg_16(transformed_frame)
    accident_proba, hidden_state = rnn_model(features, hidden_state, train = False)
    plt.imshow(frame)
    plt.text(0.95, 0.01, str(accident_proba),
            verticalalignment='bottom', horizontalalignment='right',
            # transform=plt.transAxes,
            color='red', fontsize=15)

    plt.pause(1)
    if accident_proba > 0.90:
        print('Accident Happened' )
        break
