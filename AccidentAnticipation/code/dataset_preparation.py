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


def findFrames(clip, directory_path, clip_path, crash_distance):
    start_number = (int)(clip['start'])
    crash_number = (int)(clip['crash'])
    probabilities = []

    for subdirs, dirs, files in os.walk(directory_path):

        list.sort(files)
        start_pos, crash_pos = findPositions(start_number, crash_number, files)

        print(start_pos, crash_pos)
        for position in range(start_pos, crash_pos + 1):
            shutil.copy(directory_path + files[position], clip_path)

            if (crash_pos - position) < crash_distance:
                probabilities.append(1)
            else:
                probabilities.append(0)

    return probabilities


def storeClip(clip, directory_path, storage_path, groundtruth_path, clip_number, crash_distance):
    clip_dir = 'clip_' + str(clip_number) + '/'
    clip_path = storage_path + clip_dir
    os.mkdir(clip_path)

    clip_probs = np.array(findFrames(clip, directory_path, clip_path, crash_distance))

    groundtruth_file = groundtruth_path + 'clip_' + str(clip_number) + '.txt'
    np.savetxt(groundtruth_file, clip_probs)


def main():
    path = "../data/images/"

    # Modify for threshold of accident since frame
    crash_distance = 10

    dataset_dir = '../dataset/'
    train_dir = dataset_dir + 'train/'
    test_dir = dataset_dir + 'test/'

    videoclips_dir = 'videoclips/'
    groundtruths_dir = 'groundtruth/'

    os.mkdir(dataset_dir)
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    os.mkdir(train_dir + videoclips_dir)
    os.mkdir(test_dir + videoclips_dir)
    os.mkdir(train_dir + groundtruths_dir)
    os.mkdir(test_dir + groundtruths_dir)

    all_clips = []
    train_clip_number = 0
    test_clip_number = 0
    for subdirs, dirs, files in os.walk(path):
        for directory in dirs:
            directory_path = path + directory + "/"
            clips = parseAnnotation(directory_path + directory + ".txt")

            for clip_number in range(len(clips)):
                # Train or test (70:30)
                if clip_number > int(0.7 * len(clips)):
                    test_clip_number += 1
                    train_test_dir = test_dir
                    clip_count = test_clip_number
                else:
                    train_clip_number += 1
                    train_test_dir = train_dir
                    clip_count = train_clip_number

                video_dir = train_test_dir + videoclips_dir
                probabilities_dir = train_test_dir + groundtruths_dir

                # Store clip frames and groundtruths
                storeClip(clips[clip_number], directory_path, video_dir, probabilities_dir, clip_count, crash_distance)

                print(clips[clip_number])

            all_clips.append(clips)

    print("Train clips stored : %s" % train_clip_number)
    print("Test clips stored : %s" % test_clip_number)


if __name__ == '__main__':
    main()