import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import models, transforms

import config


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 6)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

class FACENN():
    def __init__(self):
        #initialise model from pretrainded resnet
        
        #change number of outputs to our number of classes
        self.model = model
        #image preprocessing
        self.data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def prediction(self, picture):
        picture = picture.convert('RGB')
        #torch (and almost all ml lib) works with batches
        vect = [self.data_transforms(i) for i in [picture]]
        data = torch.utils.data.DataLoader(vect, batch_size=4, shuffle=True, num_workers=1)
        for i in data:
            image = i
        #create torch tensor
        image_tensor = Variable(image)
        #result of model(image_tensor) is a tensor too, we need to transform it to python array
        result =  self.model(image_tensor).data.cpu().tolist()[0]
        #normalisation
        result = np.array(result) - min(result)
        return result
