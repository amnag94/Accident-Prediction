# Author : Ketan Kokane <kk7471@rit.edu>
from mrcnn.config import Config


CAR = 1
TRUCK = 1
BIKE = 1
PEDESTRIAN = 1
BG = 1


class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "object_cfg"
	NUM_CLASSES = BG + CAR + TRUCK + PEDESTRIAN + BIKE
	# simplify GPU config
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1