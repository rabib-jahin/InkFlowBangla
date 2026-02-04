import cv2
import os

# Input image path
input_image = "D:\\PREV\\DeepLearningProject\\Dataset\\Dataset\\62\\Words\\62_2\\62_2_3_1.jpg"

# Output directory
output_dir = "blurred_outputs"
os.makedirs(output_dir, exist_ok=True)

# Read image (BGR)
img = cv2.imread(input_image)

if img is None:
    raise FileNotFoundError("Image not found!")

# Blur kernel sizes (must be odd)
blur_levels = [3, 7, 11, 15, 21]

for k in blur_levels:
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    
    # Output filename
    output_path = os.path.join(output_dir, f"blur_k{k}.jpg")
    
    # Save image
    cv2.imwrite(output_path, blurred)
    
    print(f"Saved: {output_path}")

print("All blurred images saved successfully.")
