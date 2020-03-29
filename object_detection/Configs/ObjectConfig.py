# Author : Ketan Kokane <kk7471@rit.edu>

from mrcnn.config import Config


class ObjectConfig(Config):
    NAME = "object_cfg"
    NUM_CLASSES = 3
    STEPS_PER_EPOCH = 196
