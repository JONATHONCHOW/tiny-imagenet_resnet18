import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

model = models.__dict__['resnet18']()
input_num = model.fc.in_features
model.fc = nn.Linear(input_num, 200)
# gpu to cpu
summary(model, (3, 64, 64), device = "cpu")