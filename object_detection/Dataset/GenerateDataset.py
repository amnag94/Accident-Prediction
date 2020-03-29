# Author : Ketan Kokane <kk7471@rit.edu>

import os
import shutil

from Dataset.Constants import Annotation


def merge_two_dictionary(car, not_car):
    updates = 0
    both = {}
    for key in car:
        if key in not_car:
            not_car[key].extend(car[key])
            both[key] = car[key] + not_car[key]
            updates += 1
    print(updates)
    return both


def parse_annotations():
    # have to parse all these files
    car = {}
    not_car = {}
    car_count = 0
    not_car_count = 0
    objects = ['Car', 'Pedestrian', 'Bike', 'Truck']
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
                            box_cordinate = eval(frame.split(':')[1].strip())
                            img = file.split('.')[0] + '/' + img + '.jpg'
                            if obj == 'Car':
                                add_to_dectionary(car, box_cordinate, img,
                                                  'Car')
                            else:
                                add_to_dectionary(not_car, box_cordinate, img,
                                                  'Object But Not Car')

    return car, not_car


def add_to_dectionary(_dec, box_cordinate, img, obj):
    if img in _dec:
        _dec[img].append(Annotation(obj, box_cordinate))
    else:
        _dec[img] = [Annotation(obj, box_cordinate)]


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
    for key in sortedKeys:
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
    make_image_folders('images/', 'train/')
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


def generate_test_set(training_set, car, not_car):
    test_set = {}

    for dec in [car, not_car]:
        idx = 1
        for key in dec.keys():
            if idx > 50:
                break
            if key not in training_set.keys():
                test_set[key] = dec[key]
                idx += 1
    print(len(test_set))
    return test_set


if __name__ == '__main__':
    make_dirs()
    car, not_car = parse_annotations()
    training_set = merge_two_dictionary(car, not_car)
    copy_files(training_set, 'train')
    print(len(training_set))
    print(training_set)
    testing_set = generate_test_set(training_set, car, not_car)
    copy_files(testing_set, 'test')
