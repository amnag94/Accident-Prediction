# Author : Ketan Kokane <kk7471@rit.edu>
from ObjectConfig import ObjectConfig
from ObjectDataset import ObjectDataset
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
from mrcnn.utils import compute_ap
import numpy as np


def evaluate_model(dataset, model, cfg):
    APs = list()
    for image_id in dataset.image_ids:
        # load image, bounding boxes and masks for the image id
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(
            dataset, cfg, image_id, use_mini_mask=False)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)
        # extract results for first sample
        r = yhat[0]
        # calculate statistics, including AP
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"],
                                 r["class_ids"], r["scores"], r['masks'])
        # store
        APs.append(AP)
    # calculate the mean AP across all images
    mAP = np.mean(APs)
    return mAP

def main():

    config = ObjectConfig()
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    model.load_weights('mrcnn/mask_rcnn_coco.h5', by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    train_set = ObjectDataset()
    train_set.load_dataset('')
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))
    # train weights (output layers or 'heads')

    model.train(train_set, train_set, learning_rate=config.LEARNING_RATE, epochs=5,
                layers='heads')


if __name__ == '__main__':
    main()