import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt

from models.classification.dnn import SimpleDNN
from models.classification.cnn import CustomCNN
from models.classification import resnet


class StickerAttack:
    def __init__(
        self,
        model,
        device="cpu",
        mask_path="masks/mask.png",
        printable_colors=None,
        lambda_reg=0.01,
        lambda_nps=0.1,
        num_iterations=500,
        learning_rate=1e-2,
        perturb_bound=0.3
    ):
        """
        Class to perform sticker-based adversarial attacks.

        Args:
            model:            A PyTorch model (in eval mode).
            device:           'cpu' or 'cuda'.
            mask_path:        Path to the mask image (grayscale).
            printable_colors: Tensor of shape [K, 3], each row in [0,1].
            lambda_reg:       Regularization coefficient for perturbation magnitude.
            lambda_nps:       Coefficient for non-printability score.
            num_iterations:   Number of optimization steps.
            learning_rate:    Learning rate for the optimizer.
            perturb_bound:    Clamping bound for perturbation (±).
        """
        self.model = model
        self.device = device
        self.mask = self._load_mask(mask_path)  # shape: [H, W]
        self.mask = self.mask.unsqueeze(0).unsqueeze(0)  # => [1,1,H,W]
        self.mask = self.mask.to(self.device)

        # Default set of printable colors if none is provided
        if printable_colors is None:
            self.printable_colors = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], device=self.device)
        else:
            self.printable_colors = printable_colors.to(self.device)

        self.lambda_reg = lambda_reg
        self.lambda_nps = lambda_nps
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.perturb_bound = perturb_bound
        self.criterion = nn.CrossEntropyLoss()

    def _load_mask(self, mask_path):
        """
        Loads a grayscale mask from disk and converts it to a binary 0/1 tensor.
        """
        image = Image.open(mask_path).convert('L')  # grayscale
        image_array = np.array(image)
        # Convert to binary (1 where > 128, else 0)
        binary_array = (image_array > 128)
        return torch.tensor(binary_array)

    def random_transform(self, image):
        """
        Apply a random affine transform and brightness jitter to simulate physical conditions.
        """
        angle = random.uniform(-10, 10)  # degrees
        translate = (
            random.uniform(-0.1, 0.1) * image.size(2),
            random.uniform(-0.1, 0.1) * image.size(3)
        )
        scale = random.uniform(0.9, 1.1)
        shear = random.uniform(-5, 5)

        transformed = TF.affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear
        )
        brightness_factor = random.uniform(0.8, 1.2)
        transformed = TF.adjust_brightness(transformed, brightness_factor)
        return transformed

    def nps_loss(self, perturbation):
        """
        Non-printability score: sum of distances from each pixel in the
        perturbation to the closest color in self.printable_colors.
        """
        # perturbation: [1, 3, H, W]
        pert = perturbation.view(3, -1).transpose(0, 1)  # => [H*W, 3]
        # Distances to each printable color
        distances = torch.cdist(pert, self.printable_colors)  # => [H*W, K]
        min_dist, _ = torch.min(distances, dim=1)
        return torch.sum(min_dist)

    def attack_single_image(self, x, target_label):
        """
        Runs the optimization loop on a single image x (shape [1, 3, H, W])
        with a given target_label (shape [1]).
        Returns the final adversarial image.
        """
        # Initialize delta
        delta = torch.zeros_like(x, requires_grad=True, device=self.device)
        optimizer = optim.Adam(
            [delta],
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        for i in range(self.num_iterations):
            optimizer.zero_grad()

            # Sticker constraint: multiply by self.mask
            # Here, we expand mask to [1,3,H,W] so the shape matches
            mask_3ch = self.mask.expand_as(x[:, 0:1, :, :])  # => [1,1,H,W]
            mask_3ch = mask_3ch.repeat(1, 3, 1, 1)            # => [1,3,H,W]
            perturbation = mask_3ch * delta

            # Adversarial image
            adv_image = torch.clamp(x + perturbation, 0.0, 1.0)

            # Random physical transform
            transformed_adv = self.random_transform(adv_image)

            # Forward pass
            logits = self.model(transformed_adv)
            cls_loss = self.criterion(logits, target_label)

            # L2 norm of perturbation
            reg_loss = torch.norm(perturbation, p=2)

            # Non-printability loss
            np_loss = self.nps_loss(perturbation)

            # Combine
            loss = cls_loss + self.lambda_reg * reg_loss + self.lambda_nps * np_loss
            loss.backward()
            optimizer.step()

            # Clamp delta
            with torch.no_grad():
                delta.clamp_(-self.perturb_bound, self.perturb_bound)

            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.num_iterations} "
                      f"Total Loss={loss.item():.4f} "
                      f"cls_loss={cls_loss.item():.4f} "
                      f"reg_loss={reg_loss.item():.4f} "
                      f"np_loss={np_loss.item():.4f}")

        # Return final adversarial image
        with torch.no_grad():
            final_perturbation = (mask_3ch * delta).detach()
            adv_image_final = torch.clamp(x + final_perturbation, 0.0, 1.0)
        return adv_image_final


# -----------------------
#   ATTACK SUCCESS RATE
# -----------------------
def compute_and_print_attack_success_rate(
    model,
    clean_images,
    adv_images,
    ground_truths,
    target_label,
    device=None
):
    """
    Prints the stationary (lab) test attack success rate for a targeted attack.
    """
    correct_count = 0
    success_count = 0

    for i in range(len(clean_images)):
        clean_img = clean_images[i].unsqueeze(0)
        adv_img = adv_images[i].unsqueeze(0)
        true_lbl = ground_truths[i]

        if device:
            clean_img = clean_img.to(device)
            adv_img = adv_img.to(device)
            true_lbl = torch.tensor([true_lbl], device=device)

        with torch.no_grad():
            pred_clean = model(clean_img).argmax(dim=1)
            pred_adv = model(adv_img).argmax(dim=1)

        # Check if clean image was classified correctly
        if pred_clean.item() == true_lbl.item():
            correct_count += 1
            # Check if adversarial image is predicted as target_label
            if pred_adv.item() == target_label:
                success_count += 1

    if correct_count == 0:
        asr = 0.0
    else:
        asr = success_count / correct_count

    print(f"Attack success rate: {asr:.2%}")

def preprocess_image(img_path, device):
    """
    Opens an image from disk and applies the same transformations
    used by your model. Returns a 4D tensor [1, C, H, W].
    """
    pil_img = Image.open(img_path).convert("RGB")
    # Example: resize to 224x224 and convert to tensor
    pil_img = pil_img.resize((224, 224))
    img_tensor = TF.to_tensor(pil_img).unsqueeze(0)  # shape [1,3,224,224]
    return img_tensor.to(device)


def main():
    parser = argparse.ArgumentParser(description="Generate adversarial examples (Sticker Attack)")
    parser.add_argument("--model_type", type=str, choices=["cnn", "resnet"], required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_classes", type=int, default=43)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_samples", type=int, default=15, help="Number of images to process")
    parser.add_argument("--target_label", type=int, default=5, help="Target class for sticker attack")
    parser.add_argument("--lambda_reg", type=float, default=0.01, help="Coefficient for L2 regularization")
    parser.add_argument("--lambda_nps", type=float, default=0.1, help="Coefficient for non-printability score")
    parser.add_argument("--num_iterations", type=int, default=500, help="Number of optimization steps")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--perturb_bound", type=float, default=0.3, help="Clamp bound for perturbations")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Load model
    model = load_model(args.model_type, args.num_classes, device)

    # 2) Initialize the StickerAttack object
    attack = StickerAttack(
        model=model,
        device=device,
        mask_path="masks/mask.png",  # Adjust path as needed
        lambda_reg=args.lambda_reg,
        lambda_nps=args.lambda_nps,
        num_iterations=args.num_iterations,
        learning_rate=args.learning_rate,
        perturb_bound=args.perturb_bound
    )

    # 3) Choose sample images using the CSV file (reference snippet approach)
    df = pd.read_csv(LABELS_PATH, sep=";")
    sample_data = df.sample(n=args.num_samples, random_state=42)
    # This CSV typically has columns like: 'Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'
    # We'll assume 'Filename' and 'ClassId' are present.

    clean_images = []
    adv_images = []
    ground_truths = []

    for idx, row in sample_data.iterrows():
        filename = row["Filename"]
        true_label = row["ClassId"]
        img_path = os.path.join(DATASET_PATH, filename)

        # Preprocess the image
        x = preprocess_image(img_path, device)  # shape [1,3,224,224]

        # Attack the image
        target_label_tensor = torch.tensor([args.target_label], device=device)
        adv_x = attack.attack_single_image(x, target_label_tensor)

        # Store for evaluation
        clean_images.append(x.squeeze(0).cpu())  # shape [3,224,224] on CPU
        adv_images.append(adv_x.squeeze(0).cpu())
        ground_truths.append(true_label)

        # Save adversarial image
        out_name = f"adv_{filename}"
        out_path = os.path.join(args.output_dir, out_name)
        adv_np = adv_x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        adv_pil = Image.fromarray((adv_np * 255).astype(np.uint8))
        adv_pil.save(out_path)
        print(f"Saved adversarial image => {out_path}")

    # 4) Compute Attack Success Rate
    compute_and_print_attack_success_rate(
        model=model,
        clean_images=clean_images,
        adv_images=adv_images,
        ground_truths=ground_truths,
        target_label=args.target_label,
        device=device
    )




if __name__ == "__main__":
    main()
