# Author : Ketan Kokane <kk7471@rit.edu>
from ObjectDataset import ObjectDataset
from PredictionConfig import PredictionConfig
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
import  numpy as np
from mrcnn.utils import compute_ap


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
    config = PredictionConfig()

    model = MaskRCNN(mode='inference', model_dir='./', config=config)
    # my model
    model.load_weights('mrcnn/mask_rcnn_coco.h5', by_name=True)

    test_set = ObjectDataset()
    test_set.load_dataset('')
    test_set.prepare()

    test_mAP = evaluate_model(test_set, model, config)
    print("Test mAP: %.3f" % test_mAP)



if __name__ == '__main__':
    main()
