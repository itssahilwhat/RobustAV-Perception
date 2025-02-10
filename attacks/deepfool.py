import foolbox as fb
import numpy as np
import torch

def perform_deepfool_attack(model, data_loader, steps=50, device="cuda"):
    model = model.eval().to(device)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), preprocessing=None)

    adversarial_images = []
    labels = []
    success_count = 0

    for images, true_labels in data_loader:
        images, true_labels = images.to(device), true_labels.to(device)
        attack = fb.attacks.L2DeepFoolAttack()
        raw, clipped, success = attack(fmodel, images, true_labels, steps=steps)

        adversarial_images.append(clipped.cpu().numpy())
        labels.append(true_labels.cpu().numpy())
        success_count += success.sum().item()

    success_rate = success_count / len(data_loader.dataset)
    print(f"DeepFool Attack Success Rate: {success_rate:.2f}")

    return np.concatenate(adversarial_images), np.concatenate(labels)
