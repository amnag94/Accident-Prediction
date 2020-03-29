# Author : Ketan Kokane <kk7471@rit.edu>
import numpy as np
import skimage
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from Configs.PredictionConfig import PredictionConfig
from Dataset.Constants import TRAINED_MODEL_PATH
from mrcnn.model import MaskRCNN, mold_image


# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(image, model, cfg):
    # convert pixel values (e.g. center)
    scaled_image = mold_image(image, cfg)
    # convert image into one sample
    sample = np.expand_dims(scaled_image, 0)
    # make prediction
    yhat = model.detect(sample, verbose=0)[0]

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


def load_image(image_path):
    image = skimage.io.imread(image_path)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


if __name__ == '__main__':
    config = PredictionConfig()
    model = MaskRCNN(mode='inference', model_dir='./', config=config)
    # my model
    model.load_weights(TRAINED_MODEL_PATH, by_name=True)

    image_name = '002210.jpg'
    image = load_image('../image/' + image_name)
    plot_actual_vs_predicted(image, model, config)
