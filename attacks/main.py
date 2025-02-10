import os
import argparse
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import foolbox as fb
from foolbox.attacks import LinfPGD, FGSM, DeepFoolAttack
from models.classification.cnn import CustomCNN
from models.classification.dnn import SimpleDNN
from models.classification import resnet


# Load weights and model
def load_model(model_type, num_classes, weights_path, device):
    if model_type == "dnn":
        input_size = 128 * 128 * 3
        model = SimpleDNN(input_size=input_size, num_classes=num_classes)
    elif model_type == "cnn":
        model = CustomCNN(num_classes=num_classes)
    elif model_type == "resnet":
        model = resnet.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Invalid model type")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval().to(device)
    return model



# Load random samples from the dataset
def load_random_samples(data_dir, num_samples):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = [dataset[i] for i in indices]
    return samples, dataset.classes


def generate_adversarial_examples(model, images, labels, attack_type, output_dir, device):
    fmodel = fb.PyTorchModel(model, bounds=(-1, 1), device=device)
    attacks = {
        "fgsm": FGSM(),
        "pgd": LinfPGD(),
        "deepfool": DeepFoolAttack(),
    }

    if attack_type not in attacks:
        raise ValueError(f"Invalid attack type: {attack_type}. Choose from: {list(attacks.keys())}")

    attack = attacks[attack_type]

    for i, (image, label) in enumerate(zip(images, labels)):
        image = image.unsqueeze(0).to(device)
        label = torch.tensor([label]).to(device)

        # Generate adversarial example
        _, clipped, is_adv = attack(fmodel, image, label, epsilons=0.03)

        # Save adversarial example
        adversarial_image = clipped.squeeze(0).cpu().numpy()
        adversarial_image = np.transpose(adversarial_image, (1, 2, 0))  # Convert to HWC
        adversarial_image = ((adversarial_image + 1) * 127.5).astype(np.uint8)  # Rescale to [0, 255]
        output_path = os.path.join(output_dir, f"{attack_type}_adversarial_{i}.png")
        Image.fromarray(adversarial_image).save(output_path)
        print(f"Adversarial example saved at: {output_path}")


    # Display image (optional)
    Image.fromarray(adversarial_image).show()
    
    # Calculate attack success rate
    success_rate = is_adv.float().mean().item() * 100
    print(f"Attack success rate: {success_rate:.2f}%")



def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Foolbox")
    parser.add_argument("--model_type", type=str, choices=["dnn", "cnn", "resnet"], required=True, help="Model type to attack")
    parser.add_argument("--weights", type=str, required=True, help="Path to the model weights file")
    parser.add_argument("--image", type=str, required=True, help="Path to a sample image to attack")
    parser.add_argument("--label", type=int, required=True, help="True label of the image")
    parser.add_argument("--attack", type=str, choices=["fgsm", "pgd", "deepfool"], required=True, help="Type of adversarial attack")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save adversarial examples")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of classes in the model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: 'cuda' or 'cpu'")
    args = parser.parse_args()

    # Prepare directories
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model and weights
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, args.num_classes, args.weights, device)

    # Load sample image
    images = load_random_samples(args.image).to(device)

    # Convert label to tensor
    labels = torch.tensor([args.label]).to(device)

    # Generate adversarial example
    generate_adversarial_examples(model, images, labels, args.attack, os.path.join(args.output_dir, args.attack), device)


if __name__ == "__main__":
    main()



