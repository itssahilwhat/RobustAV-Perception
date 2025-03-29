import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Enable cuDNN benchmarking for optimized GPU performance
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

DATA_DIR = "data"

# Define the ConvAutoencoder (must match your defense code)
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # outputs in [-1, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def load_gtsrb_data(data_dir, batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Use existing train/test directories
    train_dataset = datasets.GTSRB(
        root=os.path.join(data_dir, "train"),
        split="train",
        transform=transform,
        download=False  # Set to False to avoid re-downloading
    )
    val_dataset = datasets.GTSRB(
        root=os.path.join(data_dir, "test"),
        split="test",
        transform=transform,
        download=False
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

# --- Perceptual Loss using VGG16 ---
class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        # Use features from conv3_3 layer (index 16)
        self.vgg_features = nn.Sequential(*list(vgg.children())[:17])
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        out_features = self.vgg_features(output)
        target_features = self.vgg_features(target)
        return self.criterion(out_features, target_features)

# --- Simple FGSM-like attack for adversarial training ---
def fgsm_attack_autoencoder(x, epsilon=0.05):
    perturbed = x + epsilon * torch.sign(x)
    perturbed = torch.clamp(perturbed, -1, 1)
    return perturbed

# --- PGD-based attack for adversarial training ---
def pgd_attack_autoencoder(x, autoencoder, epsilon=0.05, alpha=0.01, num_iter=10):
    perturbed = x.clone().detach().requires_grad_(True)
    for _ in range(num_iter):
        outputs = autoencoder(perturbed)
        loss = nn.MSELoss()(outputs, x)
        autoencoder.zero_grad()
        loss.backward()
        with torch.no_grad():
            perturbed = perturbed + alpha * perturbed.grad.sign()
            perturbed = x + torch.clamp(perturbed - x, min=-epsilon, max=epsilon)
            perturbed = torch.clamp(perturbed, -1, 1)
        perturbed = perturbed.detach().clone().requires_grad_(True)
    return perturbed.detach()

def train_autoencoder(epochs=15, batch_size=32, device="cuda", data_dir="data", adv_training=False, adv_method="fgsm"):
    os.makedirs(os.path.join("models", "gtsrb"), exist_ok=True)
    train_loader, val_loader = load_gtsrb_data(data_dir, batch_size)
    autoencoder = ConvAutoencoder().to(device)
    mse_loss = nn.MSELoss()
    perceptual_loss_fn = PerceptualLoss(device)
    optimizer = optim.AdamW(autoencoder.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Use automatic mixed precision (AMP) if available
    scaler = torch.amp.GradScaler() if device == "cuda" else None
    device_type = "cuda" if device == "cuda" else "cpu"

    print("\n🚀 Training Autoencoder with Hybrid Loss" +
          (" + Adversarial Training" if adv_training else "") +
          f" using {adv_method.upper()} attack...")
    for epoch in range(epochs):
        autoencoder.train()
        total_loss = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, enabled=(scaler is not None)):
                outputs_clean = autoencoder(images)
                loss_clean = mse_loss(outputs_clean, images) + 0.1 * perceptual_loss_fn(outputs_clean, images)
                if adv_training:
                    if adv_method.lower() == "pgd":
                        adv_images = pgd_attack_autoencoder(images, autoencoder, epsilon=0.05, alpha=0.01, num_iter=10)
                    elif adv_method.lower() == "both":
                        adv_images_fgsm = fgsm_attack_autoencoder(images, epsilon=0.05)
                        adv_images_pgd = pgd_attack_autoencoder(images, autoencoder, epsilon=0.05, alpha=0.01, num_iter=10)
                        adv_images = 0.5 * adv_images_fgsm + 0.5 * adv_images_pgd
                    else:  # default to FGSM
                        adv_images = fgsm_attack_autoencoder(images, epsilon=0.05)
                    outputs_adv = autoencoder(adv_images)
                    loss_adv = mse_loss(outputs_adv, images) + 0.1 * perceptual_loss_fn(outputs_adv, images)
                    loss = 0.5 * loss_clean + 0.5 * loss_adv
                else:
                    loss = loss_clean
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device_type, enabled=(scaler is not None)):
                    outputs = autoencoder(images)
                    loss = mse_loss(outputs, images) + 0.1 * perceptual_loss_fn(outputs, images)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if device == "cuda":
            torch.cuda.empty_cache()

    model_save_path = os.path.join("AVs Adv Attack", "models", "gtsrb", "autoencoder.pth")
    torch.save(autoencoder.state_dict(), model_save_path)
    print("\n✅ Autoencoder trained and saved successfully at:", model_save_path)

def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder on GTSRB with Hybrid Loss (and optional adversarial training)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", type=str, default="data", help="Root directory for GTSRB data (with 'train' and 'test' subdirs)")
    parser.add_argument("--adv_training", action="store_true", help="Enable adversarial training for the autoencoder")
    parser.add_argument("--adv_method", type=str, default="fgsm", choices=["fgsm", "pgd", "both"],
                        help="Adversarial attack method to use for adversarial training (FGSM, PGD, or both)")
    args = parser.parse_args()

    train_autoencoder(epochs=args.epochs, batch_size=args.batch_size, device=args.device,
                      data_dir=args.data_dir, adv_training=args.adv_training, adv_method=args.adv_method)

if __name__ == "__main__":
    main()