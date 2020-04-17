# code design 

## VGG_16.py
holds everything that has to deal with extracting (1 * 4096) features from an image (Yes, an image and not the entire video clips)

When executed as a single script, it downloads the model implemented by pytorch amends it by removin the last 6 layers. 
and stores it, at a given path. 

* Contains a function(`get_features_tensors(image)`), which takes in an image and returns the features extracted from it.


TODD: Everyone who wants to use this, needs to know the path of this model, (I want to change this)
*TO Use need to know the path where the model is saved*
> All the paths are relative from the current file location


## RNN.py
This file is just supposed to hold the model definitation of the entire RNN model, 
like how many layers, and the forward function. 

Cost function is not the part of this, as it suits more for train file which is responsible for using RNN.py to train the model 
using part of the entire dataset, and store it at a path(the model storing can again be of some path)


*If need to change the layers of the model, need to change this file*
this file would change very rarely

## TrainRNN.py
* Use the function `get_features_tensors_video_clip(video_clip)` from VGG_16 to get the features from the entire video clips(it basically returns the features for every image), here the batch size is different

## PredicdRnn.py
* use the function `get_features_tensors(image)` from VGG_16 to get the features from the image

## TestRNN.py
* use the function `get_features_tensors(image)` from VGG_16 to get the features from the image

All the above mentioned file needs a pipeline function (module, which takes in an image)


## How to read in the video clips and annotations

Need a function, which would read a video clip (given a path) and also the ground truth
* Then need a function which would generate the required paths, 
* The predict function does not need to deal with the annotations. ( but just the video clip )



### Functions


