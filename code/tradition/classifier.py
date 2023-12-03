import argparse

import torch
import torchvision.transforms as transforms

from PIL import Image

#Efficient model from https://github.com/lukemelas/EfficientNet-PyTorch/tree/master
from efficientnet_pytorch.model import EfficientNet

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--modelName", help= "model name", default = "efficientnet-b4")
parser.add_argument("-m", "--modelPath", help= "model parameters path", default = r"..\..\result\EfficientNet-B4\trainComplete.parm")
parser.add_argument("-i", "--Image", help= "image path", default = r"C:\Users\User\Desktop\DSC_0006.JPG")

args = parser.parse_args()

modelName = args.modelName
modelPath = args.modelPath
imagePath = args.Image

tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

usingModel = EfficientNet.from_pretrained(modelName, num_classes = 525)
usingModel.to(device)

usingModel.load_state_dict(torch.load(modelPath, map_location=device))
usingModel.eval()


img = Image.open(imagePath)
img = tfms(img).unsqueeze(0)

predict = torch.argmax(usingModel(img), dim=1).to("cpu").numpy()[0]

print(predict)