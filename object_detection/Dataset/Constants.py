# Author : Ketan Kokane <kk7471@rit.edu>
from dataclasses import dataclass

GENERATED_FILE_PATH = '../../generated_data/'
TRAINED_MODEL_PATH = 'mask_rcnn_object_cfg_0001_proper.h5'

@dataclass
class Annotation:
    label: str
    box: list
