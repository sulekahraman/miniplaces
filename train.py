import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark=True

import dataset
from models.AlexNet import *
from models.ResNet import *

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res


# def acc1(data_loader, model, device):
#     correct = 0
#     total = 0
#     with torch.no_grad():            
#         for data in data_loader:
#             inputs, labels = data
#             inputs = inputs.to(device) #get the inputs and labels
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     acc = 100 * correct / total
#     err = 1 - acc
#     print('Accuracy of the network on the training data: %d %%' % (acc))
#     return acc


def validate(val_loader, model, criterion, device):
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
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            total_acc1 += acc1[0].item()
            total_acc5 += acc5[0].item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f acc1: %.4f acc5: %.4f ' % (
                    epoch, 
                    batch_num*1.0/num_train_batches,
                    running_loss/output_period,
                    acc1,
                    acc5
                    # total_acc1/num_train_batches,
                    # total_acc5/num_train_batches
                    ))
                running_loss = 0.0
                gc.collect()
    top1 = total_acc1/num_val_batches
    top5 = total_acc5/num_val_batches
    print("Accuracies on validation set: Top-1: " + str(top1) + " Top-5: " + str(top5))
    return (top1, top5)
   


def train(train_loader, model, criterion, optimizer, epoch, device):

    output_period = 100
    batch_size = 100
    num_train_batches = len(train_loader)
    total_acc1 = 0.0
    total_acc5 = 0.0
    running_loss = 0.0
    for param_group in optimizer.param_groups:
        print('Current learning rate: ' + str(param_group['lr']))
    model.train()

    for batch_num, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device) #get the inputs and labels
        labels = labels.to(device)

        optimizer.zero_grad() #zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        total_acc1 += acc1[0].item()
        total_acc5 += acc5[0].item()


        if batch_num % output_period == 0:
            print('[%d:%.2f] loss: %.3f acc1: %.4f acc5: %.4f ' % (
                epoch, 
                batch_num*1.0/num_train_batches,
                running_loss/output_period,
                acc1,
                acc5
                # total_acc1/num_train_batches,
                # total_acc5/num_train_batches
                ))
            running_loss = 0.0
            gc.collect()

#top-1 score, you check if the top class (the one having the highest probability) is the same as the target label.

#top-5 score, you check if the target label is one of your top 5 predictions (the 5 ones with the highest probabilities).

#the top score is computed as the times a predicted label matched the target label, divided by the number of data-points evaluated.      

    top1 = total_acc1/num_train_batches
    top5 = total_acc5/num_train_batches

    return (top1, top5)


def run():
    # Parameters
    num_epochs = 10
    output_period = 100
    batch_size = 100

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_18()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    # num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # TODO: optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    epoch = 1
    while epoch <= num_epochs:
        train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch, device)
        val_top1, val_top5 = validate(val_loader, model, criterion, device)
        # running_loss = 0.0
        # for param_group in optimizer.param_groups:
        #     print('Current learning rate: ' + str(param_group['lr']))
        # model.train()

        # for batch_num, (inputs, labels) in enumerate(train_loader, 1):
        #     inputs = inputs.to(device) #get the inputs and labels
        #     labels = labels.to(device)

        #     optimizer.zero_grad() #zero the parameter gradients
        #     outputs = model(inputs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()

        #     optimizer.step()
        #     running_loss += loss.item()

        #     if batch_num % output_period == 0:
        #         print('[%d:%.2f] loss: %.3f' % (
        #             epoch, batch_num*1.0/num_train_batches,
        #             running_loss/output_period
        #             ))
        #         running_loss = 0.0
        #         gc.collect()
        print("Epoch: ", epoch)

        print("Training Top-1 Accuracy: ", train_top1)
        print("Training Top-5 Accuracy: ", train_top5)

        print("Validation Top-1 Accuracy: ", val_top1)
        print("Validation Top-5 Accuracy: ", val_top5)
        print("--------------------------------")
        gc.collect()
        # save after every epoch
        torch.save(model.state_dict(), "models/model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
       
        # Training Set
        # train_accuracy1 = acc1(train_loader, model, device)
        # train_accuracy1 = accuracy(output, target, topk=(1,)):
        # train_accuracy5 = acc5(train_loader, model, device)

        # #Validation Set
        # val_accuracy1 = acc1(val_loader, model, device)
        # val_accuracy5 = acc5(val_loader, model, device)

        


        gc.collect()
        epoch += 1

print('Starting training')
run()
print('Training terminated')
