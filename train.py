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

import json


loss_ep = dict()


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


def validate(val_loader, model, criterion, device, epoch):
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
            total_acc1 += acc1.item()
            total_acc5 += acc5.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f acc1: %.4f acc5: %.4f ' % (
                    epoch, 
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
   


def train(train_loader, model, criterion, optimizer, epoch, device):

    output_period = 100
    batch_size = 100
    num_train_batches = len(train_loader)
    total_acc1 = 0.0
    total_acc5 = 0.0
    running_loss = 0.0
    loss_ep[epoch] = 0.0 
    for param_group in optimizer.param_groups:
        print('Current learning rate: ' + str(param_group['lr']))
    model.train() #comment this out if you're training sth new
    

    for batch_num, (inputs, labels) in enumerate(train_loader, 1):
        inputs = inputs.to(device) #get the inputs and labels
        labels = labels.to(device)

        optimizer.zero_grad() #zero the parameter gradients
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc1, acc5 = accuracy(outputs.data, labels.data, topk=(1, 5))
        total_acc1 += acc1.item()
        total_acc5 += acc5.item()


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
            loss_ep[epoch] += running_loss

            running_loss = 0.0
            gc.collect()

#top-1 score, you check if the top class (the one having the highest probability) is the same as the target label.

#top-5 score, you check if the target label is one of your top 5 predictions (the 5 ones with the highest probabilities).

#the top score is computed as the times a predicted label matched the target label, divided by the number of data-points evaluated.      
    top1 = total_acc1*1.0/num_train_batches
    top5 = total_acc5*1.0/num_train_batches
    loss_ep[epoch] = loss_ep[epoch]/num_train_batches

    return (top1, top5)

def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] *= gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']

def run():
    # Parameters
    num_epochs = 30
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

    # can input a weight decay argument here, shouldn't be very large since we have a large dataset , try (1e-3)
    # also try to change the learning rate  
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay = 5e-4)  #since adam is faster, might be better for lower epochs 
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer,lambda x:x*0.1)
    #scheduler takes optimizer as arguemnt, scheduler.step()
    #simple multistep scheduler , 150 epochs, drop lr at 50, and 100,multiply lr by 0.1 , increase learning rate to something like 0.1
    #5e-4 for weight decay, or 1e-4
    #increase amount of epochs  to ~30 , 20 and 25 for dropping learing rate 
    #people use 0.9, safe value of momentum 

    train_t1 = dict()
    train_t5 = dict()
    val_t1 = dict()
    val_t5 = dict()


    epoch = 1
    while epoch <= num_epochs:
        # load pre-trained model
        # Comment out the following line if you're training sth new!!
        # if epoch == 20 or epoch == 25:
        #     scheduler.step()
        #model.load_state_dict(torch.load("models/model." + str(epoch)))
        model = model.to(device)
        train_top1, train_top5 = train(train_loader, model, criterion, optimizer, epoch, device)
        val_top1, val_top5 = validate(val_loader, model, criterion, device, epoch)
#huh
        print("Epoch: ", epoch)
        print("Training Top-1 Accuracy: ", train_top1)
        print("Training Top-5 Accuracy: ", train_top5)
        print("Validation Top-1 Accuracy: ", val_top1)
        print("Validation Top-5 Accuracy: ", val_top5)
        print("--------------------------------")
        #save the errors
        train_t1[epoch] = 100 - train_top1
        train_t5[epoch] = 100 - train_top5
        val_t1[epoch] = 100 - val_top1
        val_t5[epoch] = 100 - val_top5

        # save after every epoch
        torch.save(model.state_dict(), "models/SGD_SCHEDULER_model.%d" % epoch)

        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
       
        gc.collect()
        epoch += 1

    # with open('output/dropout/train_top1.json', 'w') as out1:
    #     json.dump(train_t1, out1)
    with open('scheduler/train_top5.json', 'w') as out2:
        json.dump(train_t5, out2)
    with open('scheduler/val_top5.json', 'w') as out3:
        json.dump(val_t5, out3)
    # with open('output/dropout/val_top1.json', 'w') as out4:
    #     json.dump(val_t1, out4)
    # with open('output/dropout/loss.json', 'w') as out5:
    #     json.dump(loss_ep, out5)

print('Starting training')
run()
print('Training terminated')
