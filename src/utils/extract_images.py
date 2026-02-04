import pickle
import os
from PIL import Image
from tqdm import tqdm

def save_images_per_writer(pickle_path, output_dir, num_writers=10, imgs_per_writer=10):
    """
    Saves 'imgs_per_writer' images for the first 'num_writers' found in the dataset.
    """
    
    # 1. Create Output Directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading pickle from: {pickle_path}...")
    try:
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        return

    # 2. Get the correct dataset dictionary
    # Structure is assumed to be data['train'][writer_id] = [list of samples]
    if 'train' in data_dict:
        dataset = data_dict['train']
    else:
        dataset = data_dict

    writers_processed = 0
    total_images_saved = 0

    print(f"Extracting {imgs_per_writer} images for {num_writers} writers...")

    # 3. Iterate through writers
    # sorting keys ensures deterministic order (optional, but good for consistency)
    writer_ids = sorted(list(dataset.keys()))

    for writer_id in tqdm(writer_ids):
        if writers_processed >= num_writers:
            break

        samples = dataset[writer_id]
        
        # Skip writers that don't have enough samples (optional check)
        if len(samples) < 1: 
            continue

        # Create a subfolder for this writer to keep things organized
        writer_dir = os.path.join(output_dir, f"writer_{writer_id}")
        os.makedirs(writer_dir, exist_ok=True)

        saved_count = 0
        
        for idx, sample in enumerate(samples):
            if saved_count >= imgs_per_writer:
                break

            # Extract Image
            if isinstance(sample, dict):
                img = sample.get('img')
            else:
                img = sample

            if img is None:
                continue

            # Ensure PIL
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(img)
                except:
                    continue

            # Save
            filename = f"w{writer_id}_{idx}.png"
            save_path = os.path.join(writer_dir, filename)
            img.save(save_path)
            
            saved_count += 1
            total_images_saved += 1

        writers_processed += 1

    print(f"\nDone! Saved {total_images_saved} images across {writers_processed} writers in '{output_dir}'.")

if __name__ == "__main__":
    # CONFIGURATION
    PICKLE_FILE = 'BN-UNIFIED-NO-SINGLE.pickle' 
    OUTPUT_FOLDER = './extracted_images'
    
    save_images_per_writer(PICKLE_FILE, OUTPUT_FOLDER, num_writers=100, imgs_per_writer=10)