# Author : Ketan Kokane <kk7471@rit.edu>
import numpy as np

from ParseTextFile import formDatabase
from mrcnn.utils import Dataset


class ObjectDataset(Dataset):

    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "Car")
        self.add_class("dataset", 2, "Truck")
        self.add_class("dataset", 3, "Pedestrian")
        self.add_class("dataset", 4, "Bike")


        image_dict = formDatabase('train')

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
        # parse the annotations
        # how to assign box and label ?
        boxes = []
        labels = []
        for an in annotations:
            labels.append(an.label)
            boxes.append(an.box)
        return boxes, labels


if __name__ == '__main__':
    train_set = ObjectDataset()
    train_set.load_dataset('')
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
