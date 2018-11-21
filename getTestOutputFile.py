from __future__ import print_function
from __future__ import division
import csv
import torch
import torchvision.transforms as transforms
import pandas as pd
import os

from scipy.misc import imread, imresize

from models.AlexNet import *
from models.ResNet import *


def load_model(model_name):
    """load the pre-trained model"""
    if model_name == 'ResNet':
        model = resnet_34()
    elif model_name == 'AlexNet':
        model = alexnet()
    else:
        raise NotImplementedError(model_name + ' is not implemented here')

    modelName = "Adam_resnet_34_lr_1e-3.1"
    model.load_state_dict(torch.load("models/"+modelName,map_location='cpu'))
    return model


def construct_transformer():
    """construct transformer for images"""
    mean = [0.45486851, 0.43632515, 0.40461355]
    std = [0.26440552, 0.26142306, 0.27963778]
    transformer = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(144),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transformer

def load_image(img_path,device):
    # load the image
    image = imread(img_path)
    transformer=construct_transformer()
    image = transformer(image)
    image = image.view(1, 3, 128, 128)
    image = image.to(device)
    return image

def main():
    def top_5(image, model):
        # run the forward process
        prediction = model(image)
        prediction = prediction.to('cpu')
        _, cls = torch.max(prediction, dim=1)
        _, pred = prediction.topk(5, 1, True, True)
        top5 = pred[0].data.cpu().numpy()
        return top5

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device=torch.device('cpu')

    # load model and set to evaluation mode
    model = load_model('ResNet')
    model.to(device)
    model.eval()

    # set image transformer
    transformer = construct_transformer()

    testData_path='data/test/999'
    img_names=[]
    for filename in os.listdir(testData_path):
        img_names.append(filename)
    img_names.sort()

    text_file = 'test.txt'

    try:
        with open(text_file, 'w') as textfile:
            #writer = txt.DictWriter(textfile, fieldnames=None)
            #writer.writeheader()
            for image_name in img_names:
                image = load_image(testData_path + '/' + image_name, device)
                top5 = top_5(image, model)
                textfile.write("%s %s %s %s %s %s\n" % ('test/'+image_name, top5[0], top5[1], top5[2], top5[3], top5[4]))
    except IOError:
        print("I/O error")






main()
