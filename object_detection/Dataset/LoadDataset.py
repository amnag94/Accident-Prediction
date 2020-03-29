# Author : Ketan Kokane <kk7471@rit.edu>


import os

from Dataset.Constants import GENERATED_FILE_PATH, Annotation


def get_dectionary_from_annotations(tt):
    Annotation('l', [])
    dec = {}
    print(GENERATED_FILE_PATH + 'annotations/' +
          tt)
    print(os.listdir('.'))
    for subdir, dirs, files in os.walk(GENERATED_FILE_PATH + 'annotations/' +
                                       tt):
        for _dir in dirs:
            for _subdir, _dirs, _files in os.walk(subdir + os.sep + _dir):
                for _file in _files:
                    with open(_subdir + os.sep + _file) as annotation_file:
                        list_of_annotations = eval(annotation_file.readline(

                        ).strip())
                        file_path = _subdir + os.sep + _file
                        file_path = file_path.replace('annotations',
                                                      'images').replace('.txt',
                                                                        '.jpg')
                        dec[file_path] = list_of_annotations
    return dec


if __name__ == '__main__':
    print(get_dectionary_from_annotations('train'))
