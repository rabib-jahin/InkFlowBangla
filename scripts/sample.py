import os
import pickle
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

# 1. ক্র্যাশ ফিক্স (OMP Error আটকানোর জন্য)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def save_real_images(pickle_path, output_folder, num_samples=5000):
    print(f"📂 Loading dataset from {pickle_path}...")
    
    # ফোল্ডার তৈরি
    os.makedirs(output_folder, exist_ok=True)
    
    # ডেটা লোড
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # সব স্যাম্পল একসাথে করা
    all_samples = []
    if 'train' in data: all_samples.extend(data['train'].values())
    if 'test' in data: all_samples.extend(data['test'].values())
    
    # লিস্ট ফ্ল্যাট করা (Flatten list of lists)
    flat_samples = [item for sublist in all_samples for item in sublist]
    
    print(f"📝 Total images found: {len(flat_samples)}")
    print(f"⬇️ Saving {num_samples} images to '{output_folder}'...")

    # লুপ চালিয়ে সেভ করা
    count = 0
    for sample in tqdm(flat_samples):
        if count >= num_samples: break
            
        try:
            img = sample['img']
            if img.mode != 'RGB': img = img.convert('RGB')

            # সাইজ ঠিক করা (আপনার জেনারেটেড ইমেজের মতো 64x256 করা)
            # এটি না করলে FID ভুল আসবে
            target_h, target_w = 64, 256
            w, h = img.size
            
            # Resize Logic
            if h != target_h:
                scale = target_h / h
                new_w = int(w * scale)
                img = img.resize((new_w, target_h), Image.LANCZOS)
            
            # Pad Logic
            if img.size[0] < target_w:
                img = ImageOps.pad(img, (target_w, target_h), color='white')
            else:
                img = img.crop((0, 0, target_w, target_h))

            # সেভ
            img.save(f"{output_folder}/real_{count}.png")
            count += 1
            
        except Exception as e:
            continue

    print("✅ Extraction Done!")

# রান করার অংশ
if __name__ == "__main__":
    # আপনার পাথগুলো এখানে দিন
    PICKLE_FILE = "BN-UNIFIED-NO-SINGLE.pickle"  
    OUTPUT_FOLDER = "./real_images_fid"       
    
    save_real_images(PICKLE_FILE, OUTPUT_FOLDER, num_samples=50000)