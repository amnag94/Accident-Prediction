# Author : Ketan Kokane <kk7471@rit.edu>
from mrcnn.config import Config


class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "object_cfg"
    NUM_CLASSES = 3
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
