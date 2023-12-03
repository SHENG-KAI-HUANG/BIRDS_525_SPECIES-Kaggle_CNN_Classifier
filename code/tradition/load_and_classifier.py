import torch
import torchvision.transforms as transforms

from PIL import Image

#Efficient model from https://github.com/lukemelas/EfficientNet-PyTorch/tree/master
from efficientnet_pytorch.model import EfficientNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelPath = r"..\..\result\EfficientNet-B4\trainComplete.parm"
imagePath = r"C:\Users\User\Desktop\images.jpg"

tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

usingModel = EfficientNet.from_pretrained('efficientnet-b4', num_classes = 525)
usingModel.to(device)

usingModel.load_state_dict(torch.load(modelPath, map_location='cpu'))
usingModel.eval()


img = Image.open(imagePath)
img = tfms(img).unsqueeze(0)

predict = torch.argmax(usingModel(img), dim=1).to("cpu").numpy()[0]

print(predict)