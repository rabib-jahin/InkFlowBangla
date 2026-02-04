import torch
import os
import random
import argparse
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def load_image(path, transform):
    try:
        img = Image.open(path).convert('RGB')
        return transform(img).unsqueeze(0) # Add batch dimension [1, 3, H, W]
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def calculate_diversity_score(gen_folder, device, num_pairs=2000):
    print("=" * 60)
    print("🌈 Calculating Diversity Score (Intra-LPIPS)")
    print(f"   Folder: {gen_folder}")
    print("=" * 60)

    # 1. Initialize LPIPS Metric
    # AlexNet is standard for LPIPS
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)

    # 2. Prepare File Pairs
    files = sorted([f for f in os.listdir(gen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(files) < 2:
        print("❌ Error: Not enough images to calculate diversity (Need at least 2).")
        return

    # Shuffle files to compare random generated images against each other
    random.shuffle(files)
    
    # Create pairs (Image A vs Image B)
    # We take the first half and compare it with the second half
    mid_point = len(files) // 2
    list_a = files[:mid_point]
    list_b = files[mid_point : mid_point + len(list_a)] # Ensure same length
    
    # Limit number of pairs if needed to save time
    if len(list_a) > num_pairs:
        list_a = list_a[:num_pairs]
        list_b = list_b[:num_pairs]

    print(f"📝 Computing distance between {len(list_a)} pairs of generated images...")

    # 3. Transforms (Standardize size)
    transform = transforms.Compose([
        transforms.Resize((64, 256)), # Same size as your generation
        transforms.ToTensor(),        # [0, 1] range
    ])

    scores = []

    # 4. Calculation Loop
    for i in tqdm(range(len(list_a)), desc="Measuring Diversity"):
        path_a = os.path.join(gen_folder, list_a[i])
        path_b = os.path.join(gen_folder, list_b[i])

        img_a = load_image(path_a, transform)
        img_b = load_image(path_b, transform)

        if img_a is None or img_b is None:
            continue

        img_a = img_a.to(device)
        img_b = img_b.to(device)

        with torch.no_grad():
            # Calculate distance between two generated images
            dist = lpips_metric(img_a, img_b)
            scores.append(dist.item())

    # 5. Final Stats
    if not scores:
        print("❌ No scores calculated.")
        return

    avg_diversity = np.mean(scores)
    std_diversity = np.std(scores)

    print("\n" + "="*40)
    print(f"✨ Diversity Score (Avg LPIPS): {avg_diversity:.5f} ± {std_diversity:.5f}")
    print("-" * 40)
    print("Interpretation:")
    print(" * HIGHER is Better (Means images are different from each other)")
    print(" * Too Low (< 0.05) -> Mode Collapse (All images look same)")
    print(" * Good Range -> 0.30 - 0.50 (for handwriting)")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_folder', type=str, required=True, help="Path to generated images")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_pairs', type=int, default=2000, help="Max pairs to evaluate")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    calculate_diversity_score(args.gen_folder, args.device, args.num_pairs)