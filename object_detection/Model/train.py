# Author : Ketan Kokane <kk7471@rit.edu>

from Configs.ObjectConfig import ObjectConfig
from Dataset.ObjectDataset import ObjectDataset
from mrcnn.model import MaskRCNN


def main():
    config = ObjectConfig()
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    model.load_weights('../mrcnn/mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    train_set = ObjectDataset()
    train_set.load_dataset('train')
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    model.train(train_set, train_set, learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


if __name__ == '__main__':
    main()
