import torch
import numpy as np
from PIL import Image, ImageOps, ImageChops
import argparse
import os
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel
import torch.nn as nn
from torch.nn import DataParallel
import copy
import string
import cv2  # Added for cropping

# Import your custom modules
from src.models.unet import UNetModel
from src.models.feature_extractor import ImageEncoder

# Configuration
OUTPUT_MAX_LEN = 95
IMG_WIDTH = 256
IMG_HEIGHT = 64

# Bengali character classes
c_classes = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঁংঃািীুূৃেৈোৌ্ৎ০১২৩৪৫৬৭৮৯!"#&\'()*+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}

def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()
tokens = {"PAD_TOKEN": char_classes}
num_tokens = len(tokens.keys())
vocab_size = char_classes + num_tokens

def label_padding(labels, num_tokens):
    ll = [letter2index.get(i, 0) for i in labels]
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if num > 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)
    return ll

def preprocess_image(img, fixed_size=(64, 256)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_width, img_height = img.size
    target_height, target_width = fixed_size
    
    if img_height != target_height:
        scale = target_height / img_height
        new_width = int(img_width * scale)
        img = img.resize((new_width, target_height), Image.LANCZOS)
        img_width = new_width
    
    if img_width < target_width:
        img = ImageOps.pad(img, size=(target_width, target_height), color="white")
    elif img_width > target_width:
        left = (img_width - target_width) // 2
        img = img.crop((left, 0, left + target_width, target_height))
    
    return img

def crop_whitespace(img_np):
    """
    Crops white borders from a numpy image array.
    Assumes image is RGB with white (255,255,255) background.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Threshold: make text white (255) and bg black (0)
    # Using 200 as threshold to catch light gray anti-aliasing pixels
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find bounding box of non-zero pixels (the text)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        # If image is empty/blank, return a small white placeholder
        return np.ones((img_np.shape[0], 10, 3), dtype=np.uint8) * 255
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Optional: Add small padding (e.g. 2px) so text doesn't touch edge
    pad = 2
    h_img, w_img, _ = img_np.shape
    
    # We only crop width (x), we keep original height (64) to maintain alignment
    x_start = max(0, x - pad)
    x_end = min(w_img, x + w + pad)
    
    cropped = img_np[:, x_start:x_end, :]
    return cropped

class BengaliInference:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        print(f"🔧 Initializing inference on {self.device}")
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load tokenizer
        print("📝 Loading tokenizer...")
        if args.model_name == 'wordstylist':
            self.tokenizer = None
            self.text_encoder = None
        else:
            self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
            self.text_encoder = CanineModel.from_pretrained("google/canine-c")
            self.text_encoder = nn.DataParallel(self.text_encoder)
            self.text_encoder = self.text_encoder.to(self.device)
            self.text_encoder.eval()
        
        # Load VAE
        if args.latent:
            print("🖼️  Loading VAE...")
            self.vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
            self.vae = DataParallel(self.vae)
            self.vae = self.vae.to(self.device)
            self.vae.requires_grad_(False)
            self.vae.eval()
        else:
            self.vae = None
        
        # Load style extractor
        print("🎨 Loading style extractor...")
        self.style_extractor = ImageEncoder(
            model_name='mobilenetv2_100', 
            num_classes=0, 
            pretrained=False,
            trainable=False
        )
        
        if os.path.exists(args.style_path):
            state_dict = torch.load(args.style_path, map_location=self.device)
            model_dict = self.style_extractor.state_dict()
            matched_keys = {k: v for k, v in state_dict.items() 
                           if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(matched_keys)
            self.style_extractor.load_state_dict(model_dict, strict=False)
            print(f"✓ Loaded style extractor from {args.style_path}")
        else:
            print(f"⚠️  Style model not found at {args.style_path}")
        
        self.style_extractor = DataParallel(self.style_extractor)
        self.style_extractor = self.style_extractor.to(self.device)
        self.style_extractor.requires_grad_(False)
        self.style_extractor.eval()
        
        # Load UNet
        print("🧠 Loading UNet model...")
        
        # Infer num_classes
        num_classes = 1 
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'module.label_emb.weight' in checkpoint:
                num_classes = checkpoint['module.label_emb.weight'].shape[0]
                print(f"📊 Detected {num_classes} writer classes from checkpoint")
            elif 'label_emb.weight' in checkpoint:
                num_classes = checkpoint['label_emb.weight'].shape[0]
                print(f"📊 Detected {num_classes} writer classes from checkpoint")
        
        self.model = UNetModel(
            image_size=args.img_size,
            in_channels=args.channels,
            model_channels=args.emb_dim,
            out_channels=args.channels,
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=args.num_heads,
            num_classes=num_classes,
            context_dim=args.emb_dim,
            vocab_size=vocab_size,
            text_encoder=self.text_encoder,
            args=args
        )
        
        self.model = DataParallel(self.model)
        self.model = self.model.to(self.device)
        
        # Load trained weights
        if os.path.exists(args.model_path):
            self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model from {args.model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {args.model_path}")
        
        self.model.eval()
        
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.stable_dif_path, 
            subfolder="scheduler"
        )
        
        print("✅ Initialization complete!\n")
    
    def load_style_images(self, style_paths):
        style_images = []
        for path in style_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Style image not found: {path}")
            
            img = Image.open(path)
            img = preprocess_image(img, fixed_size=(64, 256))
            img_tensor = self.transform(img)
            style_images.append(img_tensor)
        
        while len(style_images) < 5:
            style_images.append(style_images[-1])
        style_images = style_images[:5]
        
        style_tensor = torch.stack(style_images).unsqueeze(0) 
        return style_tensor.to(self.device)
    
    def generate(self, text, style_images, writer_id=0, num_inference_steps=50):
        print(f"🖊️  Generating: '{text}' (Using dummy Writer ID: {writer_id})")
        
        with torch.no_grad():
            if self.args.model_name == 'wordstylist':
                word_embedding = label_padding(text, num_tokens)
                word_embedding = np.array(word_embedding, dtype="int64")
                text_features = torch.from_numpy(word_embedding).long().unsqueeze(0)
                text_features = text_features.to(self.device)
            else:
                text_features = self.tokenizer(
                    [text], 
                    padding="max_length", 
                    truncation=True, 
                    return_tensors="pt",
                    max_length=200
                ).to(self.device)
            
            reshaped_images = style_images.reshape(-1, 3, 64, 256)
            style_features = self.style_extractor(reshaped_images)
            
            if self.args.latent:
                x = torch.randn((1, 4, IMG_HEIGHT // 8, IMG_WIDTH // 8)).to(self.device)
            else:
                x = torch.randn((1, 3, IMG_HEIGHT, IMG_WIDTH)).to(self.device)
            
            self.noise_scheduler.set_timesteps(num_inference_steps)
            writer_id_tensor = torch.tensor([writer_id]).long().to(self.device)
            
            for time_step in self.noise_scheduler.timesteps:
                t = torch.ones(1).long().to(self.device) * time_step.item()
                noisy_residual = self.model(
                    x, t, text_features, writer_id_tensor,
                    original_images=style_images,
                    mix_rate=self.args.mix_rate,
                    style_extractor=style_features
                )
                x = self.noise_scheduler.step(noisy_residual, time_step, x).prev_sample
            
            if self.args.latent:
                latents = 1 / 0.18215 * x
                image = self.vae.module.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
            else:
                image = (x.clamp(-1, 1) + 1) / 2
            
            image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)
            
            pil_image = Image.fromarray(image)
            
        return pil_image
    
    def generate_sentence(self, sentence, style_images, writer_id=0, num_inference_steps=50, gap=20):
        """Generate full sentence with proper word spacing"""
        words = sentence.strip().split()
        word_images_np = []
        
        print(f"\n📖 Generating sentence with {len(words)} words...")
        
        for i, word in enumerate(words):
            print(f"  [{i+1}/{len(words)}] {word}")
            # Generate word (PIL)
            word_img_pil = self.generate(word, style_images, writer_id, num_inference_steps)
            word_img_np = np.array(word_img_pil)
            
            # --- CROP WHITESPACE ---
            # This trims the image to just the content, removing the huge side gaps
            cropped_word = crop_whitespace(word_img_np)
            word_images_np.append(cropped_word)
        
        # Create gap image
        gap_img = np.ones((IMG_HEIGHT, gap, 3), dtype=np.uint8) * 255
        
        # Concatenate words with gaps
        sentence_parts = []
        for w_img in word_images_np:
            sentence_parts.append(w_img)
            sentence_parts.append(gap_img)
        
        # Remove last gap if list not empty
        if sentence_parts:
            sentence_parts = sentence_parts[:-1]
        
        # Concatenate horizontally
        if sentence_parts:
            sentence_img = np.concatenate(sentence_parts, axis=1)
        else:
            # Fallback for empty sentence
            sentence_img = np.ones((IMG_HEIGHT, 100, 3), dtype=np.uint8) * 255
        
        return Image.fromarray(sentence_img)

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def resolve_style_paths(args):
    """
    Resolve style image paths from various input methods.
    Priority: --style_image_paths > (--style_id + --style_max) > --style_images
    
    Returns:
        list: List of valid style image paths
    """
    style_paths = []
    
    # Method 1: Direct image paths (HIGHEST PRIORITY - NEW)
    if args.style_image_paths:
        print("📌 Using style images from --style_image_paths")
        style_paths = args.style_image_paths
        
    # Method 2: Construct paths from writer ID and range
    elif args.style_id is not None and args.style_max is not None:
        wid = args.style_id
        print(f"📂 Constructing paths for Writer ID: {wid} (0 to {args.style_max})")
        for i in range(args.style_max + 1):
            formatted_path = os.path.join(args.data_root, f"writer_0{wid}", f"w0{wid}_{i}.png")
            style_paths.append(formatted_path)
    
    # Method 3: Legacy explicit paths (LOWEST PRIORITY)
    elif args.style_images:
        print("📌 Using style images from --style_images (legacy)")
        style_paths = args.style_images
    
    else:
        raise ValueError(
            "❌ No style images provided! Use ONE of these methods:\n"
            "  1. --style_image_paths path1.png path2.png ... (RECOMMENDED)\n"
            "  2. --style_id 11 --style_max 4 (constructs paths automatically)\n"
            "  3. --style_images path1.png path2.png ... (legacy)"
        )
    
    # Validate that all paths exist
    missing_paths = [p for p in style_paths if not os.path.exists(p)]
    if missing_paths:
        print(f"⚠️  Warning: {len(missing_paths)} style image(s) not found:")
        for p in missing_paths[:3]:  # Show first 3
            print(f"    • {p}")
        if len(missing_paths) > 3:
            print(f"    ... and {len(missing_paths) - 3} more")
    
    return style_paths

def main():
    parser = argparse.ArgumentParser(
        description='Bengali Handwriting Generation Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Method 1: Direct image paths (RECOMMENDED)
  python inference.py --model_path model.pth --text "আমার সোনার বাংলা" \\
      --style_image_paths style1.png style2.png style3.png
  
  # Method 2: Auto-construct paths from writer ID
  python inference.py --model_path model.pth --text "চন্দ্র" \\
      --style_id 11 --style_max 4 --data_root ./extracted_images
  
  # Method 3: Legacy (still supported)
  python inference.py --model_path model.pth --text "সূর্য" \\
      --style_images path/to/img1.png path/to/img2.png
        """
    )
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--style_path', type=str, default='./style_models/mixed_bengali_mobilenetv2_100.pth', help='Path to style extractor model')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5', help='Path to stable diffusion model')
    
    # Style Arguments (3 methods)
    style_group = parser.add_argument_group('Style Image Input Methods (choose ONE)')
    
    # NEW: Direct image paths (recommended)
    style_group.add_argument(
        '--style_image_paths', 
        type=str, 
        nargs='+', 
        help='[RECOMMENDED] Direct paths to style images (e.g., style1.png style2.png style3.png)'
    )
    
    # Method 2: Auto-construct from writer ID
    style_group.add_argument('--style_id', type=str, default=None, help='Writer ID for file naming (e.g., "11")')
    style_group.add_argument('--style_max', type=int, default=None, help='Highest index of style image (used with --style_id)')
    style_group.add_argument('--data_root', type=str, default='./extracted_images', help='Root directory for extracted images (used with --style_id)')
    
    # Method 3: Legacy
    style_group.add_argument('--style_images', type=str, nargs='+', help='[LEGACY] Explicit paths to style images')

    # Input
    parser.add_argument('--text', type=str, default='ঢাকায় যানজট কমাতে মেট্রোরেল অনেক সাহায্য করছে', help='Bengali text to generate')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--writer_id', type=int, default=0, help='Dummy writer ID')

    # Params
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--mode', type=str, default='sentence', choices=['word', 'sentence'])
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--gap', type=int, default=20, help='Gap between words in pixels')

    # Model Config
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🇧🇩 Bengali Handwriting Generation - Inference")
    print("=" * 60)
    
    if args.seed is not None:
        print(f"🔒 Setting fixed seed: {args.seed}")
        set_seed(args.seed)
    
    inference = BengaliInference(args)
    
    # Resolve style paths using the new unified function
    style_paths = resolve_style_paths(args)
    
    print(f"\n🎨 Loading {len(style_paths)} style reference images...")
    style_tensor = inference.load_style_images(style_paths)
    print(f"✓ Style images loaded: {style_tensor.shape}")
    
    print(f"\n🚀 Starting generation...")
    if args.mode == 'word':
        output_image = inference.generate(args.text, style_tensor, writer_id=args.writer_id, num_inference_steps=args.num_steps)
    else:
        # Pass the gap argument here
        output_image = inference.generate_sentence(args.text, style_tensor, writer_id=args.writer_id, num_inference_steps=args.num_steps, gap=args.gap)
    
    output_dir = os.path.dirname(args.output)
    if output_dir and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    output_image.save(args.output)
    
    print(f"\n✅ Generated image saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()