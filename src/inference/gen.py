import os
import pickle
import pandas as pd
from PIL import Image
from tqdm import tqdm

def create_pickle_from_folders(root_path, output_pickle_path):
    """
    Generates a pickle dataset from a specific folder structure.
    
    Structure Expected:
    root_path/
      ├── 1/
      │   ├── Words/
      │   │     └── Subfolders/
      │   │           └── image_id.png
      │   └── 1.xlsx (Cols: Id, Word)
      ├── 2/
      ...
    """
    
    # Initialize the main dictionary structure
    # Based on your previous code, the root keys are usually 'train', 'test', etc.
    # We will put everything into 'train' for now.
    dataset = {
        'train': {}
    }
    
    # 1. Get all writer folders (assuming they are named "1", "2", "101", etc.)
    if not os.path.exists(root_path):
        print(f"Error: Root path '{root_path}' does not exist.")
        return

    writer_folders = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]
    
    # Sort them to process in order (numeric sort is usually better for writer IDs)
    try:
        writer_folders.sort(key=lambda x: int(x))
    except ValueError:
        writer_folders.sort() # Fallback to string sort if non-numeric

    print(f"Found {len(writer_folders)} writer folders.")

    total_images_processed = 0

    # 2. Iterate through each writer
    for writer_id in tqdm(writer_folders, desc="Processing Writers"):
        
        writer_path = os.path.join(root_path, writer_id)
        excel_path = os.path.join(writer_path, f"{writer_id}.xlsx")
        words_root_path = os.path.join(writer_path, "Words")

        # --- A. Validation ---
        if not os.path.exists(excel_path):
            print(f"Skipping Writer {writer_id}: Excel file missing ({excel_path})")
            continue
        
        if not os.path.exists(words_root_path):
            print(f"Skipping Writer {writer_id}: 'Words' folder missing")
            continue

        # --- B. Load Excel Ground Truth ---
        try:
            # Load Excel, ensuring IDs are strings to match filenames easier
            df = pd.read_excel(excel_path, dtype={'Id': str, 'Word': str})
            
            # Create a dictionary for fast lookup: {'image_id': 'text_label'}
            # We strip whitespace just in case
            labels_map = dict(zip(df['Id'].str.strip(), df['Word'].str.strip()))
            
        except Exception as e:
            print(f"Error reading Excel for writer {writer_id}: {e}")
            continue

        writer_samples = []

        # --- C. Walk through 'Words' folder (recursive) ---
        # "Words folder contains some folders" -> os.walk handles recursion
        for root, dirs, files in os.walk(words_root_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    
                    # Get filename without extension (e.g. "word_5.png" -> "word_5")
                    file_id = os.path.splitext(file)[0]
                    
                    # Check if we have a label for this image
                    if file_id in labels_map:
                        label_text = labels_map[file_id]
                        
                        # Verify label isn't NaN or empty
                        if pd.isna(label_text) or str(label_text).strip() == "":
                            continue
                            
                        img_path = os.path.join(root, file)
                        
                        try:
                            # Load Image
                            # We convert to RGB immediately to ensure consistency and force loading into memory
                            img = Image.open(img_path).convert('RGB')
                            
                            # Add to samples list
                            writer_samples.append({
                                'img': img,
                                'label': str(label_text)
                            })
                            total_images_processed += 1
                            
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
                    else:
                        # Optional: Print if image found but no ID in Excel
                        # print(f"Warning: Image {file_id} not found in Excel for writer {writer_id}")
                        pass

        # --- D. Add to Dataset ---
        if writer_samples:
            # Ensure writer_id is stored as the correct type (int or str)
            # Your previous dataset sorted keys, so integers are usually safer if folders are numbers
            try:
                w_key = int(writer_id)
            except:
                w_key = writer_id
                
            dataset['train'][w_key] = writer_samples

    # 3. Save Pickle
    print(f"\nCompiling pickle file with {total_images_processed} total images...")
    print("This might take a moment depending on RAM...")
    
    try:
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"✅ Success! Dataset saved to: {output_pickle_path}")
    except Exception as e:
        print(f"❌ Failed to save pickle: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # Path where your downloaded folders (1, 2, 3...) are located
    RAW_DATA_ROOT = r"D:\DeepLearningProject\Dataset\Dataset" 
    
    # Where you want to save the new pickle
    OUTPUT_PICKLE = "my_new_dataset.pickle"
    
    create_pickle_from_folders(RAW_DATA_ROOT, OUTPUT_PICKLE)