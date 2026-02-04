import torch
import os
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms
from PIL import Image
import argparse
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_images_generator(folder, transform):
    """
    Generator function to load images one by one.
    This prevents loading all images into RAM at once.
    """
    files = sorted(os.listdir(folder))
    for filename in files:
        try:
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('RGB')
            yield transform(img)
        except Exception as e:
            continue

def calculate_kid_score(real_folder, gen_folder, device, subset_size=50, batch_size=32):
    print("=" * 60)
    print("📊 Calculating Kernel Inception Distance (KID)")
    print(f"   Real Images: {real_folder}")
    print(f"   Gen Images:  {gen_folder}")
    print("=" * 60)

    # 1. Initialize KID Metric
    # subset_size: KID compares subsets of images. 50 is standard for smaller datasets.
    kid_metric = KernelInceptionDistance(subset_size=subset_size).to(device)

    # 2. Define Transforms
    # KID uses InceptionV3 which expects 299x299 images.
    # IMPORTANT: Torchmetrics KID expects images as uint8 [0, 255] tensor.
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),             # Converts to [0.0, 1.0]
        lambda x: (x * 255).byte()         # Converts back to [0, 255] uint8
    ])

    # 3. Process Real Images in Batches
    print("\n🔄 Processing Real Images...")
    real_batch = []
    for img_t in tqdm(load_images_generator(real_folder, transform), desc="Real"):
        real_batch.append(img_t)
        if len(real_batch) >= batch_size:
            batch_tensor = torch.stack(real_batch).to(device)
            kid_metric.update(batch_tensor, real=True)
            real_batch = []
    # Process remaining real images
    if real_batch:
        batch_tensor = torch.stack(real_batch).to(device)
        kid_metric.update(batch_tensor, real=True)

    # 4. Process Generated Images in Batches
    print("\n🔄 Processing Generated Images...")
    gen_batch = []
    for img_t in tqdm(load_images_generator(gen_folder, transform), desc="Generated"):
        gen_batch.append(img_t)
        if len(gen_batch) >= batch_size:
            batch_tensor = torch.stack(gen_batch).to(device)
            kid_metric.update(batch_tensor, real=False)
            gen_batch = []
    # Process remaining generated images
    if gen_batch:
        batch_tensor = torch.stack(gen_batch).to(device)
        kid_metric.update(batch_tensor, real=False)

    # 5. Compute Final Score
    print("\n🧮 Computing Final Score (this may take a moment)...")
    kid_mean, kid_std = kid_metric.compute()

    print("\n" + "="*40)
    print(f"📉 KID Score: {kid_mean.item():.6f} ± {kid_std.item():.6f}")
    print("-" * 40)
    print("Interpretation:")
    print(" * Lower is Better (0.0 is perfect match)")
    print(" * Unbiased estimator (better than FID for smaller datasets)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_folder', type=str, required=True, help="Path to real images")
    parser.add_argument('--gen_folder', type=str, required=True, help="Path to generated images")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--subset_size', type=int, default=50, help="Subset size for KID calculation")
    
    args = parser.parse_args()
    
    calculate_kid_score(args.real_folder, args.gen_folder, args.device, args.subset_size)