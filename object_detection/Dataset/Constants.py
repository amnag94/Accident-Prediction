# Author : Ketan Kokane <kk7471@rit.edu>
from dataclasses import dataclass

GENERATED_FILE_PATH = '../generated_data/'


@dataclass
class Annotation:
    label: str
    box: list
