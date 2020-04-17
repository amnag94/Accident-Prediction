import numpy as np
import pandas as pd
import shutil, os

def parseAnnotation(file_annotation):
    objects = ['Car', 'Truck', 'Bike', 'Pedestrian']
    clips = []
    with open(file_annotation, "r") as annotation:
        for line in annotation:
            parts = line.split(",")
            if parts[0] not in objects:
                clips.append({'start': parts[0], 'crash': parts[1]})
    return clips


def findPositions(start, crash, files):
    start_found = False
    crash_pos = 0
    start_pos = 0

    for file_pos in range(0, len(files)):

        frame_parts = files[file_pos].split('.')
        frame_type = frame_parts[1]

        if frame_type == "jpg":
            frame_name = int(frame_parts[0])

            if frame_name == start:
                start_pos = file_pos
                start_found = True
            if frame_name == crash and start_found:
                crash_pos = file_pos

    return start_pos, crash_pos


def findFrames(clip, directory_path, clip_path):
    start_number = int(clip['start'])
    crash_number = int(clip['crash'])

    for subdirs, dirs, files in os.walk(directory_path):

        list.sort(files)
        start_pos, crash_pos = findPositions(start_number, crash_number, files)

        print(start_pos, crash_pos)
        for position in range(start_pos, crash_pos + 1):
            shutil.copy(directory_path + files[position], clip_path)


def storeClip(clip, directory_path, storage_path, clip_number):
    clip_dir = 'clip_' + str(clip_number) + '/'
    clip_path = storage_path + clip_dir
    os.mkdir(clip_path)

    findFrames(clip, directory_path, clip_path)


def main():
    path = "../data/images/"

    dataset_dir = '../dataset/'
    train_dir = dataset_dir + 'train/'
    videoclips_dir = train_dir + 'videoclips/'

    os.mkdir(dataset_dir)
    os.mkdir(train_dir)
    os.mkdir(videoclips_dir)

    all_clips = []
    clip_number = 0
    for subdirs, dirs, files in os.walk(path):
        for directory in dirs:
            directory_path = path + directory + "/"
            clips = parseAnnotation(directory_path + directory + ".txt")

            for clip in clips:
                clip_number += 1
                storeClip(clip, directory_path, videoclips_dir, clip_number)
                print(clip)

            all_clips.append(clips)

    print("Clips stored : %s" % clip_number)


if __name__ == '__main__':
    main()