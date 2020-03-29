# Author : Ketan Kokane <kk7471@rit.edu>
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from Configs.PredictionConfig import PredictionConfig
from Dataset.Constants import TRAINED_MODEL_PATH
from Dataset.ObjectDataset import ObjectDataset
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
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
    model.load_weights(TRAINED_MODEL_PATH, by_name=True)

    test_set = ObjectDataset()
    test_set.load_dataset('test')
    test_set.prepare()

    print('Test: %d' % len(test_set.image_ids))
    for i in range(10, 25, 4):
        plot_actual_vs_predicted(test_set, model, config, i)
    # test_mAP = evaluate_model(test_set, model, config)
    # print("Test mAP: %.3f" % test_mAP)


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, i):
    image = dataset.load_image(i)
    mask, _ = dataset.load_mask(i)
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

    # print(yhat)
    # # define subplot
    # pyplot.subplot(121)
    # # plot raw pixel data
    # pyplot.imshow(image)
    # pyplot.title('Actual')
    # # plot masks
    # for j in range(mask.shape[2]):
    #     pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3, interpolation='nearest')
    # # # get the context for drawing boxes
    # pyplot.subplot(122)
    # # plot raw pixel data
    pyplot.imshow(image, interpolation='nearest')
    pyplot.title('Predicted')
    ax = pyplot.gca()
    # plot each box
    color = ['', 'red', 'blue']
    for idx in range(len(yhat['rois'])):
        box = yhat['rois'][idx]
        # get coordinates
        y1, x1, y2, x2 = box
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False,
                         color=color[yhat['class_ids'][idx]])
        # draw the box
        ax.add_patch(rect)
    # # show the figure
    pyplot.show()


if __name__ == '__main__':
    main()
