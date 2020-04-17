# Author : Ketan Kokane <kk7471@rit.edu>

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

MODEL_PATH = '../trained_models/vgg.model'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VGG_16(nn.Module):
    def __init__(self, original_model):
        super(VGG_16, self).__init__()
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = nn.Sequential(
                *list(original_model.classifier.children())[:-6])

        for param in self.features.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = False

        for param in self.avgpool.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def Generate_Model():
    vgg16 = models.vgg16(pretrained=True, progress=True)
    # this will download the entire model of 536 MB, as it has 138 million learnable params
    vgg = VGG_16(vgg16)
    torch.save(vgg, MODEL_PATH)


def Test_Model():
    loaded_model = torch.load(MODEL_PATH)
    img_path = '../dataset/train/videoclips/clip_1/000017.jpg'
    input_image = Image.open(img_path)
    # the preprocess pipeline on the image
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    features = loaded_model(input_batch)
    print(features.shape)


def getModel(path ):
    loaded_model = torch.load(path)
    return loaded_model


if __name__ == '__main__':
    Generate_Model()
    Test_Model()
