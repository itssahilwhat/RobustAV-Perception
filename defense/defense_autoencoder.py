import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from models.classification.cnn import CustomCNN

# Set paths (adjust paths as needed)
CLASSIFIER_WEIGHTS_PATH = r"models/gtsrb/cnn.pth"
DATASET_PATH = r"data/test/gtsrb/GTSRB/Final_Test/Images"
LABELS_PATH = r"data/test/gtsrb/GT-final_test.csv"

# -------------------
# Define the Autoencoder (must match train_autoencoder.py architecture)
# -------------------
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

# -------------------
# Preprocessing Functions
# -------------------
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device, non_blocking=True)

def preprocess_image_from_pil(pil_img, device="cpu"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(pil_img).unsqueeze(0).to(device, non_blocking=True)

# -------------------
# Additional JPEG Compression Preprocessing
# -------------------
def jpeg_compression(image_tensor, quality=50, device="cpu"):
    inv_norm = transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])
    img_denorm = inv_norm(image_tensor.squeeze(0).cpu())
    pil_img = transforms.ToPILImage()(img_denorm.clamp(0, 1))
    temp_path = "temp.jpg"
    pil_img.save(temp_path, "JPEG", quality=quality)
    compressed = preprocess_image(temp_path, device)
    os.remove(temp_path)
    return compressed

# -------------------
# NEW: SVD Reconstruction Function
# -------------------
def svd_reconstruct(image_tensor, retention_ratio=0.7):
    # Denormalize image to [0,1]
    inv_norm = transforms.Normalize(mean=[-0.5/0.5]*3, std=[1/0.5]*3)
    img = image_tensor.squeeze(0).cpu()
    img_denorm = inv_norm(img)
    reconstructed_channels = []
    for c in range(img_denorm.size(0)):
        channel = img_denorm[c]
        U, S, V = torch.svd(channel)
        total = S.sum()
        cumulative = torch.cumsum(S, dim=0)
        k = (cumulative / total >= retention_ratio).nonzero(as_tuple=False)[0].item() + 1
        S_k = torch.diag(S[:k])
        U_k = U[:, :k]
        V_k = V[:, :k]
        channel_reconstructed = torch.mm(U_k, torch.mm(S_k, V_k.t()))
        reconstructed_channels.append(channel_reconstructed.unsqueeze(0))
    reconstructed_img = torch.cat(reconstructed_channels, dim=0)
    norm_transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    reconstructed_img = norm_transform(reconstructed_img)
    return reconstructed_img.unsqueeze(0).to(image_tensor.device)

# -------------------
# NEW: Feature Squeezing Function
# -------------------
def feature_squeeze(image_tensor, bit_depth=4):
    image = (image_tensor + 1) / 2.0
    levels = 2 ** bit_depth - 1
    squeezed = torch.round(image * levels) / levels
    squeezed = squeezed * 2 - 1
    return squeezed

# -------------------
# Diverse Transformation Ensemble
# -------------------
def apply_diverse_transformations(image, device="cpu"):
    pil_img = transforms.ToPILImage()(image.squeeze(0).cpu().detach())
    transforms_list = [
        transforms.Compose([]),
        transforms.Compose([transforms.RandomRotation(degrees=10)]),
        transforms.Compose([transforms.RandomResizedCrop(128, scale=(0.8, 1.0))]),
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)]),
        transforms.Compose([lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))])
    ]
    transformed_tensors = []
    for t in transforms_list:
        transformed_img = t(pil_img)
        transformed_tensor = preprocess_image_from_pil(transformed_img, device)
        transformed_tensors.append(transformed_tensor)
    # Also include SVD and Feature Squeezing variants:
    svd_variant = svd_reconstruct(image, retention_ratio=0.7)
    fs_variant = feature_squeeze(image, bit_depth=4)
    transformed_tensors.extend([svd_variant, fs_variant])
    return transformed_tensors

# -------------------
# Sigmoid-based Weighting Function (tuned for higher robustness)
# -------------------
def compute_weight(similarity, threshold=0.7, k=15):
    return 1 / (1 + np.exp(-k * (similarity - threshold)))

# -------------------
# Defense Prediction with Softmax Probabilities
# -------------------
def defense_prediction_with_probs(classifier, autoencoder, image, device, similarity_threshold=0.7):
    with torch.no_grad():
        output_orig = classifier(image)
        probs_orig = F.softmax(output_orig, dim=1)
    compressed = jpeg_compression(image, quality=50, device=device)
    with torch.no_grad():
        reconstructed = autoencoder(compressed)
        output_recon = classifier(reconstructed)
        probs_recon = F.softmax(output_recon, dim=1)
    cos_sim = F.cosine_similarity(probs_orig, probs_recon)
    similarity = cos_sim.item()
    print(f"Feature cosine similarity: {similarity:.4f}")
    weight_orig = compute_weight(similarity, threshold=similarity_threshold, k=15)
    weight_recon = 1 - weight_orig
    ensemble_probs = weight_orig * probs_orig + weight_recon * probs_recon
    return ensemble_probs, reconstructed, similarity

# -------------------
# Ensemble Defense Prediction with Diverse Transformations
# -------------------
def ensemble_defense_prediction(classifier, autoencoder, image, device, similarity_threshold=0.7, num_ensemble=10):
    ensemble_probs_list = []
    variants = apply_diverse_transformations(image, device)
    while len(variants) < num_ensemble:
        variants.extend(variants)
    variants = variants[:num_ensemble]
    for variant in variants:
        probs, _, _ = defense_prediction_with_probs(classifier, autoencoder, variant, device, similarity_threshold)
        ensemble_probs_list.append(probs)
    avg_probs = torch.mean(torch.stack(ensemble_probs_list), dim=0)
    final_pred = torch.argmax(avg_probs, dim=1).item()
    return final_pred, avg_probs

# -------------------
# Load the Classifier Model (CNN)
# -------------------
def load_classifier(device, num_classes=43):
    model = CustomCNN(num_classes=num_classes)
    checkpoint = torch.load(CLASSIFIER_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device)
    model.eval()
    return model

# -------------------
# Load or Train the Autoencoder
# -------------------
AUTOENCODER_WEIGHTS_PATH = r"models/gtsrb/autoencoder.pth"

def load_autoencoder(device):
    autoencoder = ConvAutoencoder().to(device)
    if os.path.exists(AUTOENCODER_WEIGHTS_PATH):
        checkpoint = torch.load(AUTOENCODER_WEIGHTS_PATH, map_location=device)
        autoencoder.load_state_dict(checkpoint)
        print("Loaded pre-trained autoencoder weights.")
    else:
        print("No pre-trained autoencoder found. You may need to train it first!")
    autoencoder.eval()
    return autoencoder

# -------------------
# Load Ground Truth Labels (Mapping by index)
# -------------------
def load_ground_truth_labels_by_index():
    ground_truth = {}
    with open(LABELS_PATH, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            filename = row["Filename"]
            index = int(os.path.splitext(filename)[0])
            ground_truth[index] = int(row["ClassId"])
    return ground_truth

# -------------------
# Main function to run defense on a set of attacked images
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Dual-Path Ensemble Defense against digital adversarial attacks on traffic signs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of images to process")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Cosine similarity threshold")
    parser.add_argument("--num_ensemble", type=int, default=10, help="Number of transformation variants for ensemble")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing adversarial images")
    parser.add_argument("--output_dir", type=str, default="defense_outputs", help="Directory to save defense outputs")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    classifier = load_classifier(device)
    autoencoder = load_autoencoder(device)
    ground_truth = load_ground_truth_labels_by_index()

    image_files = sorted(
        [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.ppm', '.jpg', '.png'))]
    )[:args.num_samples]

    os.makedirs(args.output_dir, exist_ok=True)

    predictions = []
    correct_count = 0

    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        print(f"\nProcessing image {i}: {img_path}")
        try:
            index = int(filename.split('_')[-1].split('.')[0])
        except Exception as e:
            print(f"Error extracting index from {filename}: {e}")
            index = -1

        image = preprocess_image(img_path, device)
        final_pred, avg_probs = ensemble_defense_prediction(classifier, autoencoder, image, device, args.similarity_threshold, args.num_ensemble)
        predictions.append(final_pred)

        true_label = load_ground_truth_labels_by_index().get(index, -1)
        if final_pred == true_label:
            print(f"✅ Defense successful! Prediction: {final_pred}, Ground Truth: {true_label}")
            correct_count += 1
            compressed = jpeg_compression(image, quality=50, device=device)
            with torch.no_grad():
                reconstructed = autoencoder(compressed)
            recon_img = reconstructed.squeeze(0).cpu().detach()
            recon_img = recon_img * 0.5 + 0.5  # Denormalize
            recon_img = transforms.ToPILImage()(recon_img)
            output_path = os.path.join(args.output_dir, f"defended_{filename}")
            recon_img.save(output_path)
            print(f"Defended image saved at: {output_path}")
        else:
            print(f"❌ Defense failed. Prediction: {final_pred}, Ground Truth: {true_label}")

    success_rate = (correct_count / len(image_files)) * 100
    print(f"\n✅ Defense success rate: {success_rate:.2f}% ({correct_count}/{len(image_files)} correctly classified)")
    print("\nFinal predictions for test images:")
    print(predictions)

if __name__ == "__main__":
    main()
