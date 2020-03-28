## CCN for image detection

* is there is an implementation of faster CNN pytorch

* A Image detection model predicts class of the object and its bounding box in the image

* what is the input to such a model

* For pytorch need to write a dataset class, which mainly implements `__getitem__` and `__len__` method


## How it is implemented in the article I' am reading

### Dataset:
Penn-Fudan Database for Pedestrian Detection and Segmentation
[https://www.cis.upenn.edu/~jshi/ped_html/]

The new custom dataset `__getitem__` should return:
* image: a PIL image of size(H, W)
> what is the size of the each frame on our dataset
* target: a dictionary containing the following fields
	* __boxes__(FloatTensor[N,4]) : the coordinates the N bounding boxes in [X0, Y0, X1, Y1] format, ranging from 0 to W and 0 to H.
	* __labels__(Int64Tensor[N]): a label for each bounding box
	* __image_id__(Int64Tensor[1]): an image identifier, it should be uniqueused for evaluation
	* __area__(Tensor[N]): The area of the bounding box, this is used during evaluation with the COCO metrix, to seperate the metric scores between small, medium and large boxes. 
	* __iscrowd__(UintT8Tensor[N]): instances with isCrowd=True will be ignored during evaluation



# WE ARE DOING THIS ONE

## How to Train an Object Detection Model with Keras
> predict both where the objects are in the image and what  type of objects were detected.

Mask RCNN keras model for own object detection tasks
### Using the library requires:
1. careful preparation of the dataset


#### should be able to
1. Prepare Object detection dataset ready for modelling with a R-CNN.
2. How to use transfer learning to train an object detection model on a new dataset.
3. How to evaluate a fit maks R-cNN model on a test dataset and make predictions on new photos.

* The model allows for transfer learning with top performing models trained on challenging ds like COCO.
## Mask R-CNN
* designed to predict both bounding boxes for objects as well as masks for those detected objects. 

> This tutorial, ignores masking capabilities of the model.

### Tutorial Overview:
1. Install Mask R-CNN for Keras
	1. Mask R-CNN is a sophisticated model to implement, especially as compared to a simple or even state-of-the-art deep convolutional neural network model. Instead of developing an implementation of the R-CNN or Mask R-CNN model from scratch, we can use a reliable third-party implementation built on top of the Keras deep learning framework. The best-of-breed third-party implementations of Mask R-CNN is the Mask R-CNN Project developed by Matterport. The project is open source released under a permissive license (e.g. MIT license) and the code has been widely used on a variety of projects and Kaggle competitions.
	2. git clone https://github.com/matterport/Mask_RCNN.git
	3. cd Mask_RCNN 
	4. sudo python setup.py install


2. How to prepare Dataset for object detection
	1. getting the dataset together
	2. parsing the annotations file (would be different)
	3. developing the KangarooDataset object that can be used by Mask_RCNN library
		1. mask-rcnn library requires train, validation and test datasets be managed by a mrcnn.utils.Dataset object.
		2. Define a new class which extends the mrcnn.utils.Datasets class and override two functions i. `load_mask()` ii. `image_references()`
		``` python
			train_set = KangarooDataset()
			train_set.load_dataset(...)
			train_set.prepare()
		```
		3. for load_mask() function, use bounding boxes from annotation to create mask and return it
		4. know the helper function `self.add_image()` and `self.add_class()`
	4. Testing the DS object, if image and object annotatios are generated properly

		the mask-rcnn library provides utilities for displaying images and masks. We can use some of these built-in functions to confirm that the Dataset is operating correctly

		For example, the mask-rcnn library provides the mrcnn.visualize.display_instances() function that will show a photograph with bounding boxes, masks, and class labels. This requires that the bounding boxes are extracted from the masks via the extract_bboxes() function


* parse the dataset to get  a dictionary of path to file name : bounding boxes, label

3. How to train Mask R-CNN model for kangaroo Detection 
	1. Download the model architecture and weights for the pre-fit mask RCNN model. The file is about 250MB. Download the model weights to a file with the name `mask_rcnn_coco.h5`.
	2. Define a configuration object for the model. This is a new class that extends the `mrcnn.config.Config` and defines properties both the prediction problem (such as name and the number of classes) and the algorithm for training the model (such as the learning rate)
	3. Define our model. 
		1. Create an instance of mrcnn.model.MaskRCNN class and pass the previosly created config to it.
		`model = MaskRCNN(mode='training', model_dir='./', config=config)`
	4. Load the pre-defined model arch and weights. achived by calling the `load_weights()` function of the previously created model instance.
	5. class specific output layers to be removed so that new output layers can be defined and trained. this can be done by `exlude argument` and listing all of the output layers 
	```
	# load weights (mscoco)
	model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
	```
	6. Train the model, by specifying training set, and stating the layers to train = `head`
	7. Model gets saved at every epoch 


* A mask rcnn model can be fit from scratch, although like otehr CV applictios, time can be saved ans performance can be improved by using __transfer learning__

* The mask r-cnn model pre-fit on the MS COCO object dataset can be used as starting point and then tailored to the soecific dataset, in this case, the kangaroo dataset.



4. How to Evaluate a Mask R-CNN model 

The performance of a model for an object recognition task is often evaluated using the mean absolute precision (mAP).

* we are also predicting bounding boxes so we can determine whether a bounding box prediction is good or bad on how well the predicted and actual bounding boxes overlap. This can be calculated by dividing the area of the overlap by the total area of both bounding boxes. 

5. How to detect new Kangaroos in New Photos

#### Object detection
1. where are they
2. what are their extent
3. what are they

