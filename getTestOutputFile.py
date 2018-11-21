import gc
import sys
from torch.autograd import Variable
import torch
import torchvision.transforms as transforms

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

from scipy.misc import imread, imresize




import json


modelToUse = "Adam_resnet_34_lr_1e-3.1"
modelFilePath = "models/"+modelToUse

model = resnet_34()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)

model.load_state_dict(torch.load(modelFilePath,map_location='cpu'))
# remove map_location='cpu' if running on GPU

batch_size = 100
train_loader, val_loader = dataset.get_data_loaders(batch_size)
# num_train_batches = len(train_loader)

criterion = nn.CrossEntropyLoss().to(device)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        print("correct is")
        print(correct)
        print("-------")

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(val_loader, model, criterion, device):
    output_period = 100
    batch_size = 100    
    num_val_batches = len(val_loader)
    total_acc1 = 0.0
    total_acc5 = 0.0
    running_loss = 0.0
    model.eval() #evaluate mode
    with torch.no_grad(): 
        for batch_num, (inputs, labels) in enumerate(val_loader, 1):
            inputs = inputs.to(device) #get the inputs and labels
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            acc1, acc5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            print("------")
            print("acc1")
            print(acc1)
            print(acc1.item())
            # print(torch.Tensor.item(acc1))

            print("acc5")
            print(acc5)
            print(acc5.item())
            # print(torch.Tensor.item(acc5))
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f acc1: %.4f acc5: %.4f ' % (
                    batch_num*1.0/num_val_batches,
                    running_loss/output_period,
                    acc1,
                    acc5
                    # total_acc1/num_train_batches,
                    # total_acc5/num_train_batches
                    ))
                running_loss = 0.0
                gc.collect()
    top1 = total_acc1*1.0/num_val_batches
    top5 = total_acc5*1.0/num_val_batches
    print("Accuracies on validation set: Top-1: " + str(top1) + " Top-5: " + str(top5))
    return (top1, top5)
   

val_top1, val_top5 = validate(val_loader, model, criterion, device)
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



# image = imread('data/test/999/00000001.jpg')
# transformer = construct_transformer()
# image = transformer(image)
# # TODO: change the shape of the image to (bsz, n_channel, h, w)
# # so that it can be fed into the model. You might want to use the view function.
# image = image.view(1,3,128,128) 
# image = image.to(device)
# _, cls = torch.max(prediction, dim=1)



def getTestOutputFile():
	for i in range(1,10000):
		filename = "0"*(8-len(str(i)))+str(i)+".jpg"
		file = open("data/test/999/"+filename)
		best5 = [1,2,3,4,5]
		file.close()
		lineToPrint = "test/"+filename+" "

		j=0
		while j < len(best5):
			lineToPrint += str(best5[j])
			if j != len(best5) - 1:
				lineToPrint += " "
			j+=1
		print(lineToPrint)

#getTestOutputFile()