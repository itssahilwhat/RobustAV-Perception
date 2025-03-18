import os
import argparse
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import foolbox as fb
from foolbox.attacks import LinfPGD, LinfFastGradientAttack as FGSM, L2DeepFoolAttack as DeepFoolAttack
from foolbox.attacks import L2CarliniWagnerAttack, L2BasicIterativeAttack
from models.classification.cnn import CustomCNN
from models.classification import resnet

WEIGHTS_PATH = r"/home/itssahilwhat/DataspellProjects/AVs Adv Attack/models/gtsrb/cnn.pth"
DATASET_PATH = r"/home/itssahilwhat/DataspellProjects/AVs Adv Attack/data/gtsrb/GTSRB/Final_Test/Images"
LABELS_PATH = r"/home/itssahilwhat/DataspellProjects/AVs Adv Attack/data/gtsrb/GT-final_test.csv"

def load_model(model_type, num_classes, device):
    if model_type == "cnn":
        model = CustomCNN(num_classes=num_classes)
    elif model_type == "resnet":
        model = resnet.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.to(device).eval()
    return model

def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

def generate_adversarial_examples(model, images, labels, attack_type, output_dir, device):
    fmodel = fb.PyTorchModel(model, bounds=(-0.5, 0.5), device=device)
    attacks = {
        "fgsm": FGSM(),
        "pgd": LinfPGD(),
        "deepfool": DeepFoolAttack(steps=50),
        "cw": L2CarliniWagnerAttack(),
        "bim": L2BasicIterativeAttack(steps=100)
    }
    attack = attacks.get(attack_type)
    if not attack:
        raise ValueError(f"Invalid attack type: {attack_type}")
    os.makedirs(output_dir, exist_ok=True)

    count_success = 0
    count_failure = 0
    for i, (image, label) in enumerate(zip(images, labels)):
        label_tensor = torch.tensor([label]).to(device)
        if attack_type in ["fgsm", "pgd", "bim", "mim"]:
            _, adversarial, is_adv = attack(fmodel, image, label_tensor, epsilons=0.05)
        else:
            _, adversarial, is_adv = attack(fmodel, image, label_tensor, epsilons=[0.05])
        if isinstance(adversarial, list):
            adversarial = adversarial[0]
        adv_image = np.clip(((adversarial.squeeze(0).cpu().numpy().transpose(1,2,0) + 0.5) * 255), 0, 255).astype(np.uint8)
        output_path = os.path.join(output_dir, f"{attack_type}_adversarial_{i}.jpg")
        Image.fromarray(adv_image).save(output_path, quality=95)
        print(f"Adversarial example {i} saved at: {output_path}")
        success_rate = is_adv.float().mean().item() * 100
        print(f"Attack success rate for image {i}: {success_rate:.2f}%")
        if success_rate >= 99.9:
            count_success += 1
        elif success_rate <= 0.1:
            count_failure += 1
    print(f"\nTotal images processed: {len(images)}")
    print(f"Images with 100% attack success: {count_success}")
    print(f"Images with 0% attack success: {count_failure}")

def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples using Foolbox")
    parser.add_argument("--model_type", type=str, choices=["cnn", "resnet"], required=True)
    parser.add_argument("--attack", type=str, choices=["fgsm", "pgd", "deepfool", "cw", "bim", "mim"], required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_classes", type=int, default=43)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of images to process")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, args.num_classes, device)

    df = pd.read_csv(LABELS_PATH, sep=";")
    sample_data = df.sample(n=args.num_samples)
    images = [preprocess_image(os.path.join(DATASET_PATH, row["Filename"]), device) for _, row in sample_data.iterrows()]
    labels = sample_data["ClassId"].tolist()

    generate_adversarial_examples(model, images, labels, args.attack, args.output_dir, device)

if __name__ == "__main__":
    main()