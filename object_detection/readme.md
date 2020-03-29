Step 1: Generate the dataset:
	based on the annotations made, decided to genarate dataset 
	File `GenerateDataset.py` is responsible for generating/separating images and its annotations into different files

* How to train the model

In Model folder
execute
`python3 train.py` -> this can take around 7 hours on CPU

* How to test the model

In Model folder execute
`python3 test.py` -> tests against all the images under test foolder of generated_data/images


* How to predict with just an image
open predcit.py towards the bottom change the image_name to the required image, the image needs to be in image folder

then excute

`python3 predict.py`

RED = CAR 
BLUE = NOT A CAR
