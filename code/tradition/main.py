#setting environment
import numpy as np # linear algebra

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from time import sleep
from tqdm import tqdm, trange

#Efficient model from https://github.com/lukemelas/EfficientNet-PyTorch/tree/master
from efficientnet_pytorch.model import EfficientNet

from BirdDatasetClass import birdDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#if need increase dimension => .unsqueeze(0) #unsqueeze : insert new dimension into assing axis
batch_size = 32

tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#if need increase dimension => .unsqueeze(0) #unsqueeze : insert new dimension into assing axis
batch_size = 32
pathPrefix = "./archive/"
fileName = "birds.csv"

trainSet = birdDataset(pathPrefix + fileName, pathPrefix, "Train", tfms)
train_dataloader = DataLoader(trainSet, batch_size=batch_size, shuffle=True)

validSet = birdDataset(pathPrefix + fileName, pathPrefix, "Valid", tfms)
valid_dataloader = DataLoader(validSet, batch_size=batch_size)

testSet = birdDataset(pathPrefix + fileName, pathPrefix, "Test", tfms)
test_dataloader = DataLoader(testSet, batch_size=1)

#training and validing block
lr = 0.001
momentum = 0.9
trainEpoch = 10
validFrequency = 5

#originalEfficientNet
usingModel = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 525)
usingModel.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(usingModel.parameters(), lr=lr, momentum=momentum)

maxValidAcc = 0

for epoch in range(0, trainEpoch):
    running_loss = 0.0
    accumulateCorrect = 0
    accumulateSize = 0
    
    usingModel.train()
    with tqdm(train_dataloader, unit="batch") as progressBarLoader:
        for data in progressBarLoader:
            sleep(0.01)
            progressBarLoader.set_description(f"Train Epoch {epoch}")
            
            # get the inputs; data is a list of [inputs, labels]
            imgs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = usingModel(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            correct = (torch.argmax(outputs, dim=1) == labels).sum().item()
            accuracy = correct / batch_size
            
            accumulateCorrect += correct
            accumulateSize += batch_size
                
            progressBarLoader.set_postfix(loss=running_loss, accuracy=100. * accuracy, 
                                          accumuAccuracy=100. * (accumulateCorrect / accumulateSize))
    
    if epoch > 0 and epoch % validFrequency == 0:
        accumulateCorrect = 0
        accumulateSize = 0
        running_loss = 0.0
        
        usingModel.eval()
        with torch.no_grad():
            with tqdm(valid_dataloader, unit="batch") as progressBarLoader:
                for data in progressBarLoader:
                    sleep(0.01)
                    progressBarLoader.set_description(f"Valid Epoch {epoch}")

                    # get the inputs; data is a list of [inputs, labels]
                    imgs, labels = data[0].to(device), data[1].to(device)

                    outputs = usingModel(imgs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()

                    correct = (torch.argmax(outputs,  dim=1) == labels).sum().item()
                    accuracy = correct / batch_size
                    
                    accumulateCorrect += correct
                    accumulateSize = batch_size

                    progressBarLoader.set_postfix(loss=running_loss, accuracy=100. * accuracy,
                                                  accumuAccuracy=100. * (accumulateCorrect / accumulateSize))
        if accumulateCorrect / accumulateSize > maxValidAcc:
            PATH = './bestValid.parm'
            torch.save(usingModel.state_dict(), PATH)
                    
print("training complete")
PATH = './trainComplete.parm'
torch.save(usingModel.state_dict(), PATH)

predict = []
RealLabelList = []
RealNameList = []
RealPathList = []

usingModel.eval()
with torch.no_grad():
    with tqdm(test_dataloader, unit="batch") as progressBarLoader:
        for data in progressBarLoader:
            sleep(0.01)
            img, RealLabel, RealName, RealPath = data[0].to(device), data[1], data[2], data[3]

            predict.append(torch.argmax(usingModel(img), dim=1).to("cpu").numpy()[0])
            RealLabelList.append(RealLabel.numpy()[0])
            RealNameList.append(RealName[0])
            RealPathList.append(RealPath[0])

    output = pd.DataFrame({"Ground Truth Label": RealLabelList, "Ground Truth Name" : RealNameList, 
                           "Ground Truth Image Path" : RealPathList, "Predict Label": predict})
    output.to_csv("./predictBirds.csv", index= False)
    
print("test complete")