# Author : Ketan Kokane <kk7471@rit.edu>

from mrcnn.config import Config


CAR = 1
TRUCK = 1
BIKE = 1
PEDESTRIAN = 1
BG = 1


class ObjectConfig(Config):
    NAME = "object_cfg"
    NUM_CLASSES = BG + CAR + TRUCK + PEDESTRIAN + BIKE
    STEPS_PER_EPOCH = 131