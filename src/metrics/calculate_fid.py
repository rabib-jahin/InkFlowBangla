import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse

import pickle
import torch
import random
from PIL import Image, ImageOps
from tqdm import tqdm
from pytorch_fid.fid_score import calculate_fid_given_paths

# ==========================================
# 1. Helper Functions
# ==========================================

def preprocess_image(img, fixed_size=(64, 256)):
    """Resizes and pads image to match model output size (64x256)"""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_width, img_height = img.size
    target_height, target_width = fixed_size
    
    # Resize keeping aspect ratio
    if img_height != target_height:
        scale = target_height / img_height
        new_width = int(img_width * scale)
        img = img.resize((new_width, target_height), Image.LANCZOS)
        img_width = new_width
    
    # Pad if needed
    if img_width < target_width:
        img = ImageOps.pad(img, size=(target_width, target_height), color="white")
    elif img_width > target_width:
        left = (img_width - target_width) // 2
        img = img.crop((left, 0, left + target_width, target_height))
    
    return img

def extract_real_images(pickle_path, output_folder, num_samples=1000, seed=42):
    """Extracts real images from pickle to a folder"""
    
    # Check if folder exists and has enough images
    if os.path.exists(output_folder):
        existing_files = [f for f in os.listdir(output_folder) if f.endswith(('.png', '.jpg'))]
        if len(existing_files) >= num_samples:
            print(f"✓ Found {len(existing_files)} real images in {output_folder}. Skipping extraction.")
            return

    print(f"📂 Extracting real images from {pickle_path}...")
    os.makedirs(output_folder, exist_ok=True)

    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)

    # Collect all samples
    all_samples = []
    if 'train' in data:
        for writer_id, samples in data['train'].items():
            all_samples.extend(samples)
    if 'test' in data:
        for writer_id, samples in data['test'].items():
            all_samples.extend(samples)

    print(f"  - Total available real images: {len(all_samples)}")

    # Randomly select samples
    random.seed(seed)
    # If num_samples is massive (like 1 million), cap it at len(all_samples)
    actual_num_samples = min(len(all_samples), num_samples)
    
    selected_samples = random.sample(all_samples, actual_num_samples)

    print(f"📝 Saving {len(selected_samples)} images for FID calculation...")
    
    for idx, sample in enumerate(tqdm(selected_samples, desc="Extracting")):
        img = sample['img']
        img = preprocess_image(img, fixed_size=(64, 256))
        
        save_path = os.path.join(output_folder, f"real_{idx:05d}.png")
        img.save(save_path)

    print("✓ Extraction complete!")

# ==========================================
# 2. Main FID Calculation Logic
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='Calculate FID Score')
    parser.add_argument('--real_images_pickle', type=str, required=True, help='Path to dataset pickle')
    parser.add_argument('--generated_images_folder', type=str, default='./generated_for_fid', help='Path to generated images')
    parser.add_argument('--num_real_samples', type=int, default=5000, help='Number of real images to use (Higher is better)')
    parser.add_argument('--temp_real_folder', type=str, default='./temp_real_images_fid', help='Temp folder for real images')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size for Inception model')
    
    args = parser.parse_args()

    # Step 1: Extract Real Images
    extract_real_images(
        args.real_images_pickle, 
        args.temp_real_folder, 
        num_samples=args.num_real_samples
    )

    # Step 2: Calculate FID using Library Function (No Subprocess)
    print("\n" + "="*60)
    print("🚀 Computing FID Score...")
    print(f"• Real Path:      {args.temp_real_folder}")
    print(f"• Generated Path: {args.generated_images_folder}")
    print("="*60)
    print("⏳ Loading InceptionV3 model and computing statistics...")
    print("(This step will show a progress bar for each folder)")

    try:
        fid_value = calculate_fid_given_paths(
            paths=[args.temp_real_folder, args.generated_images_folder],
            batch_size=args.batch_size,
            device=args.device,
            dims=2048, # Standard dimension for FID
            num_workers=0 # Set to 0 to avoid multiprocessing issues on Windows/some Linux setups
        )
        
        print("\n" + "-" * 30)
        print(f"🏆 FID SCORE: {fid_value:.4f}")
        print("-" * 30)
        
    except Exception as e:
        print(f"\n❌ Error computing FID: {e}")

if __name__ == "__main__":
    main()