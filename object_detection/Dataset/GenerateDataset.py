# Author : Ketan Kokane <kk7471@rit.edu>

import os
import shutil
from Dataset.Constants import Annotation


def parse_annotations():
    # have to parse all these files
    dec = {}
    objects = ['Car', 'Pedestrian']
    for subdir, dirs, files in os.walk('../../Annotations/'):
        for file in files:
            if file.endswith('.txt'):
                print(f'parsing {file}')
                with open(subdir + os.sep + file) as annotation_file:
                    for line in annotation_file:
                        if line.split(',')[0] not in objects:
                            continue
                        obj = line.split(',')[0]
                        frames = line[line.find('{') + 1: line.find('}')]
                        for frame in frames.strip().split("',"):
                            frame = frame.strip().replace("'", "")
                            img = frame.split(':')[0]
                            box_cordinate =  eval(frame.split(':')[1].strip())
                            img = file.split('.')[0] + '/' + img + '.jpg'
                            if img in dec:
                                dec[img].append(Annotation(obj, box_cordinate))
                            else:
                                dec[img] = [Annotation(obj, box_cordinate)]
    return dec


def write_annotations(data, file_name):
    with open(file_name, 'w') as file:
        file.writelines(repr(data))


def copy_files(dec, train_test):
    sortedKeys = list(dec.keys())
    sortedKeys.sort()
    train_test = train_test + '/'

    new_image_path = '../../generated_data/images/' + train_test
    annotation_file_path = '../../generated_data/annotations/' + train_test
    old_image_path = '../../data/'
    for key in sortedKeys[:100]:
        img_file = key
        annotation_file = key.split('.')[0] + '.txt'
        # print(old_image_path + img_file, new_image_path + img_file,
        #       annotation_file_path + annotation_file)
        shutil.copyfile(old_image_path + img_file, new_image_path + img_file)
        write_annotations(dec[key], annotation_file_path + annotation_file)


def make_image_folders(img_ann, train_test):
    for dir in range(41, 52):
        path = '../../generated_data/' + img_ann + train_test + '/00' + str(dir)
        os.mkdir(path)


def make_dirs():
    path = '../../generated_data'
    os.mkdir(path)
    path = '../../generated_data/images'
    os.mkdir(path)
    path = '../../generated_data/images/train'
    os.mkdir(path)
    make_image_folders('images/','train/')
    path = '../../generated_data/images/test'
    os.mkdir(path)
    make_image_folders('images/', 'test/')

    path = '../../generated_data/annotations'
    os.mkdir(path)
    path = '../../generated_data/annotations/train'
    os.mkdir(path)
    make_image_folders('annotations/', 'train/')
    path = '../../generated_data/annotations/test'
    os.mkdir(path)
    make_image_folders('annotations/', 'test/')


if __name__ == '__main__':
    make_dirs()
    dec = parse_annotations()
    print(len(dec))
    print(dec)
    #
    copy_files(dec, 'train')
    copy_files(dec, 'test')