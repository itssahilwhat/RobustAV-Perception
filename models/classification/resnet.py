from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch


def get_resnet_model(num_classes, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)