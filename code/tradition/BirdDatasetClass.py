import torch
from torch.utils.data import Dataset

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image


#define dataset class
class birdDataset(Dataset):
    def __init__(self, path, pathPrefix, runningType="T", transform=None):
        self.wrongName = ["PARAKETT  AKULET"]
        
        if runningType == "Valid":
            self.rightName = ["PARAKETT AUKLET"]
        else:
            self.rightName = ["PARAKETT  AUKLET"]
        
        fullData = pd.read_csv(path)
        if runningType == "Train":
            self.birdData = fullData.loc[fullData['data set'] == "train"]
        elif runningType == "Valid":
            self.birdData = fullData.loc[fullData['data set'] == "valid"]
        else:
            self.birdData = fullData.loc[fullData['data set'] == "test"]
        
        self.birdData.reset_index(inplace=True)
        
        self.transform = transform
        self.runningType = runningType
        self.pathPrefix = pathPrefix

    def __len__(self):
        return len(self.birdData)

    def showBirdInfo(self, img, name, label):
        display(img)
        print("it name is " + name + " with label = " + str(label))
    
    def __getitem__(self, idx):
        #clean data
        imgPath = self.pathPrefix + self.birdData.at[idx, "filepaths"];
        imgPath = imgPath.replace(self.wrongName[0], self.rightName[0]);
        
        img = Image.open(imgPath)
        label = self.birdData.at[idx, "class id"].astype('int64')
        name = self.birdData.at[idx, "labels"]
        
        #self.showBirdInfo(img, name, label)
        
        if self.transform:
            img = self.transform(img)
            
        if self.runningType == "Train":
            return img, torch.as_tensor(label)
        elif self.runningType == "Valid":
            return img, torch.as_tensor(label)
        else:
            return img, label, name, imgPath