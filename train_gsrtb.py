import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn import CustomCNN
from models.dnn import SimpleDNN
from models  import resnet
import PIL
# Dataset setup
def load_gsrtb_data(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    target_transform = transforms.Lambda(lambda y: torch.tensor(y))

    train_dataset = datasets.GTSRB(root=f"{data_dir}/train", split='train',transform=transform, target_transform=target_transform,download=True)
    val_dataset = datasets.GTSRB(root=f"{data_dir}/test",split='test', transform=transform, target_transform=target_transform, download=True)
    
 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    

    
    return train_loader, val_loader, train_dataset.num_classes

# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_acc += (preds == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = outputs.max(1)
                val_acc += (preds == labels).sum().item()
        
        val_acc /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
    return model



# Main function
def main():
    parser = argparse.ArgumentParser(description="Train a model on the GSRTB dataset")
    parser.add_argument('--model', type=str, choices=['dnn', 'resnet', 'cnn'], required=True, help="Model type: 'dnn', 'resnet', or 'cnn'")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()
    
    # Load data
    train_loader, val_loader, num_classes = load_gsrtb_data("data/gstrb", args.batch_size)
    
    # Initialize model
    if args.model == 'dnn':
        input_size = 128 * 128 * 3  # Assuming resized images
        model = SimpleDNN(input_size=input_size, num_classes=num_classes)
    elif args.model == 'resnet':
        model = resnet.get_resnet_model(num_classes)
    elif args.model == 'cnn':
        model = CustomCNN(num_classes=num_classes)
    else:
        raise ValueError("Invalid model type")
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=args.epochs, device=args.device)

if __name__ == "__main__":
    main()
