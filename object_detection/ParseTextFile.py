import numpy as np
from dataclasses import dataclass
import os

@dataclass
class annotation:
    label : str
    box : list

def getTextFile(file_name, path):
    return path  + "/" + file_name + ".txt"

def parseTextFile(text_file_name, file_name, path):
    with open(text_file_name) as text_file:
        dec = {}
        objects = ['Car', 'Truck', 'Pedestrian', 'Bike']
        for line in text_file:
            if line.split(',')[0] in objects:
                obj = line.split(',')[0]
                frames = line[line.find('{') + 1: line.find('}')]

                lst = []
                for frame in frames.split("',"):
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

def formDatabase(folder):
    path = 'data/' #+ folder + '/'
    files = [f.name for f in os.scandir(path) if f.is_dir()]

    images_dict = {}

    for file in files:
        text_file_name = getTextFile(file, 'Annotations/')
        dictionary = parseTextFile(text_file_name, file, path)

        for key in dictionary:
            images_dict[key] = dictionary[key]

    return images_dict


if __name__ == '__main__':
    images_dict = formDatabase('train')

    for idx, key in enumerate(list(images_dict.keys())):
        print(idx, key, images_dict[key])
