import numpy as np
from dataclasses import dataclass
import os

@dataclass
class annotation:
    label : str
    box : list

def getTextFile(file_name, path):
    return path + '/' + file_name + "/" + file_name + ".txt"

def parseTextFile(text_file_name, file_name, path):
    with open(text_file_name) as text_file:
        dec = {}
        objects = ['Car', 'Truck', 'Pedestrian', 'Bike']
        for line in text_file:
            if line.split(',')[0] in objects:
                obj = line.split(',')[0]
                frames = line[line.find('{') + 1: line.find('}')]

                for frame in frames.split("',"):
                    lst = []

                    frame = frame.strip().replace("'", "")
                    img = frame.split(':')[0]
                    coor = frame.split(':')[1]
                    positions = coor[coor.find('[') + 1: coor.find(']')].split(',')

                    for index in range(len(positions)):
                        positions[index] = int(positions[index])

                    key = path + file_name + '/' + img + '.jpg'
                    if key in dec:
                        lst = dec[key]
                    lst.append(annotation(obj, positions))
                    dec[key] = lst

    return dec

def formDatabase():
    path = 'data/images/' #+ folder + '/'
    files = [f.name for f in os.scandir(path) if f.is_dir()]

    images_dict = {}

    for file in files:
        text_file_name = getTextFile(file, path)
        dictionary = parseTextFile(text_file_name, file, path)

        for key in dictionary:
            images_dict[key] = dictionary[key]

    return images_dict, path

def formTrainTestDb():

    images_dict, path = formDatabase()

    objects_cars = ['Car', 'Bike', 'Truck']

    object_dict = {}
    for key in images_dict:
        for item in images_dict[key]:
            lst = []

            if item.label in objects_cars:
                label = path + 'Cars'
            else:
                label = path + 'Pedestrians'

            if label in object_dict:
                lst = object_dict[label]

            lst.append(item)
            object_dict[label] = lst
            #print("Key : %s " % key + "Label : %s " % item.label + "Boxes : %s" % item.box)

    train_test_dict = {}

    lst_cars = object_dict[path + 'Cars']
    lst_pedestrian = object_dict[path + 'Pedestrians']

    cars_train_part = int(0.7 * len(lst_cars))
    pedestrain_train_part = int(0.7 * len(lst_pedestrian))

    train_test_dict[path + 'train/Cars'] = lst_cars[:cars_train_part]
    train_test_dict[path + 'train/Pedestrians'] = lst_pedestrian[:pedestrain_train_part]

    train_test_dict[path + 'test/Cars'] = lst_cars[cars_train_part + 1:]
    train_test_dict[path + 'test/Pedestrians'] = lst_pedestrian[pedestrain_train_part + 1:]

    print(train_test_dict)

    #print(len(train_test_dict[path + 'train/Cars']), len(train_test_dict[path + 'train/Pedestrians']))
    #print(len(train_test_dict[path + 'test/Cars']), len(train_test_dict[path + 'test/Pedestrians']))

    return train_test_dict

if __name__ == '__main__':
    formTrainTestDb()