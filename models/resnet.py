from torchvision.models import resnet18
import torch
import torch.nn as nn
# Modify ResNet for custom classes
def get_resnet_model(num_classes):
    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Replace last layer
    return model
    