import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.classification.cnn import CustomCNN
from models.classification.dnn import SimpleDNN
from models.classification import resnet


def load_gtsrb_data(data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.GTSRB(root=f"{data_dir}/train", split='train', transform=transform, download=True)
    val_dataset = datasets.GTSRB(root=f"{data_dir}/test", split='test', transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, 43


def train_model(model, train_loader, val_loader, num_epochs=5, device='cuda', checkpoint_path=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = train_correct / len(train_loader.dataset)

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_acc = val_correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['dnn', 'resnet', 'cnn'], required=True)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs("models/gtsrb", exist_ok=True)

    train_loader, val_loader, num_classes = load_gtsrb_data("data/", args.batch_size)

    if args.model == 'dnn':
        model = SimpleDNN(input_size=128 * 128 * 3, num_classes=num_classes)
    elif args.model == 'resnet':
        model = resnet.get_resnet_model(num_classes)
    else:
        model = CustomCNN(num_classes=num_classes)

    checkpoint_path = f"models/gtsrb/{args.model}.pth"
    trained_model = train_model(model, train_loader, val_loader, args.epochs, args.device, checkpoint_path)


if __name__ == "__main__":
    main()