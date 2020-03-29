# Author : Ketan Kokane <kk7471@rit.edu>
import numpy as np

from Dataset.LoadDataset import *
from mrcnn.utils import Dataset
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances


class ObjectDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "Car")
        self.add_class("dataset", 2, "Object But Not Car")

        image_dict = get_dectionary_from_annotations(dataset_dir)

        for idx, key in enumerate(list(image_dict.keys())):
            self.add_image('dataset', image_id=idx, path=key,
                           annotation=image_dict[key])

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotation = info['annotation']
        h, w = 720, 1280
        boxes, labels = self.extract_boxes(annotation)
        masks = np.zeros([h, w, len(boxes)], dtype='uint8')
        class_ids = list()

        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index(labels[i]))

        return masks, np.asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def extract_boxes(self, annotations):
        boxes = []
        labels = []
        for an in annotations:
            labels.append(an.label)
            boxes.append(an.box)
        return boxes, labels


if __name__ == '__main__':
    train_set = ObjectDataset()
    train_set.load_dataset('train')
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # define image id
    image_id = 1
    # load the image
    image = train_set.load_image(image_id)
    # load the masks and the class ids
    mask, class_ids = train_set.load_mask(image_id)
    # extract bounding boxes from the masks
    bbox = extract_bboxes(mask)
    # display image with masks and bounding boxes
    display_instances(image, bbox, mask, class_ids, train_set.class_names)
