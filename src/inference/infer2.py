import torch
import numpy as np
from PIL import Image, ImageOps
import argparse
import os
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
import torch.nn as nn
from torch.nn import DataParallel
import cv2

# Import your custom modules
from src.models.unet import UNetModel
from src.models.feature_extractor import ImageEncoder
from tokenizer import BengaliGraphemeTokenizer

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

def preprocess_image(img, fixed_size=(64, 256)):
    """Preprocess image to fixed size"""
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
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    
    if coords is None:
        return np.ones((img_np.shape[0], 10, 3), dtype=np.uint8) * 255
        
    x, y, w, h = cv2.boundingRect(coords)
    
    pad = 2
    h_img, w_img, _ = img_np.shape
    
    x_start = max(0, x - pad)
    x_end = min(w_img, x + w + pad)
    
    cropped = img_np[:, x_start:x_end, :]
    return cropped

class BengaliGraphemeInference:
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
        print("📝 Loading Bengali Grapheme Tokenizer...")
        self.tokenizer = BengaliGraphemeTokenizer(output_max_len=OUTPUT_MAX_LEN)
        
        # First, try to infer vocab size from checkpoint
        vocab_size_from_checkpoint = None
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'module.token_embedding.weight' in checkpoint:
                vocab_size_from_checkpoint = checkpoint['module.token_embedding.weight'].shape[0]
                print(f"📊 Detected vocabulary size from checkpoint: {vocab_size_from_checkpoint}")
            elif 'token_embedding.weight' in checkpoint:
                vocab_size_from_checkpoint = checkpoint['token_embedding.weight'].shape[0]
                print(f"📊 Detected vocabulary size from checkpoint: {vocab_size_from_checkpoint}")
        
        # Build vocab from sample texts (or load pre-built vocab if available)
        if args.vocab_texts_file and os.path.exists(args.vocab_texts_file):
            print(f"📚 Building vocabulary from {args.vocab_texts_file}...")
            with open(args.vocab_texts_file, 'r', encoding='utf-8') as f:
                vocab_texts = [line.strip() for line in f if line.strip()]
            self.tokenizer.build_vocab(vocab_texts)
        elif args.pickle_path and os.path.exists(args.pickle_path):
            print(f"📚 Building vocabulary from training pickle file: {args.pickle_path}...")
            import pickle
            with open(args.pickle_path, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Extract all texts from the dataset
            all_texts = []
            for writer_id, samples in data_dict['train'].items():
                for sample in samples:
                    all_texts.append(sample['label'])
            
            print(f"   Found {len(all_texts)} text samples")
            self.tokenizer.build_vocab(all_texts)
        else:
            print("⚠️ No vocab file or pickle provided, using minimal vocab")
            sample_texts = ['অআইঈউঊঋএঐওঔ', 'কখগঘঙচছজঝঞ', 'টঠডঢণতথদধন', 
                          'পফবভমযরলশষসহ', '০১২৩৪৫৬৭৮৯']
            self.tokenizer.build_vocab(sample_texts)
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        print(f"🔢 Vocabulary size: {self.vocab_size}")
        
        # Verify vocab size matches checkpoint
        if vocab_size_from_checkpoint is not None:
            if self.vocab_size != vocab_size_from_checkpoint:
                print(f"\n⚠️ WARNING: Vocabulary size mismatch!")
                print(f"   Tokenizer vocab size: {self.vocab_size}")
                print(f"   Checkpoint expects: {vocab_size_from_checkpoint}")
                print(f"\n💡 You need to build vocab from the same dataset used in training.")
                print(f"   Try using: --pickle_path ./bengali_dataset.pickle")
                raise ValueError(
                    f"Vocabulary size mismatch: tokenizer has {self.vocab_size} tokens, "
                    f"but checkpoint expects {vocab_size_from_checkpoint} tokens"
                )
        
        self.text_encoder = None
        
        # Load VAE
        if args.latent:
            print("🖼️ Loading VAE...")
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
            print(f"⚠️ Style model not found at {args.style_path}")
        
        self.style_extractor = DataParallel(self.style_extractor)
        self.style_extractor = self.style_extractor.to(self.device)
        self.style_extractor.requires_grad_(False)
        self.style_extractor.eval()
        
        # Load UNet
        print("🧠 Loading UNet model...")
        
        # Infer num_classes from checkpoint
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
            vocab_size=self.vocab_size,
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
        """Load and preprocess style images"""
        style_images = []
        for path in style_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Style image not found: {path}")
            
            img = Image.open(path)
            img = preprocess_image(img, fixed_size=(64, 256))
            img_tensor = self.transform(img)
            style_images.append(img_tensor)
        
        # Ensure we have exactly 5 style images
        while len(style_images) < 5:
            style_images.append(style_images[-1])
        style_images = style_images[:5]
        
        style_tensor = torch.stack(style_images).unsqueeze(0)  # [1, 5, C, H, W]
        return style_tensor.to(self.device)
    
    def generate(self, text, style_images, writer_id=0, num_inference_steps=50):
        """Generate a single word/phrase"""
        print(f"🖊️ Generating: '{text}' (Writer ID: {writer_id})")
        
        with torch.no_grad():
            # Encode text using grapheme tokenizer
            encoded_indices = self.tokenizer.encode(text)
            text_features = torch.tensor(encoded_indices).long().unsqueeze(0)
            text_features = text_features.to(self.device)
            
            # Extract style features
            reshaped_images = style_images.reshape(-1, 3, 64, 256)
            style_features = self.style_extractor(reshaped_images)
            
            # Initialize noise
            if self.args.latent:
                x = torch.randn((1, 4, IMG_HEIGHT // 8, IMG_WIDTH // 8)).to(self.device)
            else:
                x = torch.randn((1, 3, IMG_HEIGHT, IMG_WIDTH)).to(self.device)
            
            # Setup scheduler
            self.noise_scheduler.set_timesteps(num_inference_steps)
            writer_id_tensor = torch.tensor([writer_id]).long().to(self.device)
            
            # Denoising loop
            for time_step in self.noise_scheduler.timesteps:
                t = torch.ones(1).long().to(self.device) * time_step.item()
                noisy_residual = self.model(
                    x, 
                    timesteps=t, 
                    context=text_features, 
                    y=writer_id_tensor,
                    style_extractor=style_features
                )
                x = self.noise_scheduler.step(noisy_residual, time_step, x).prev_sample
            
            # Decode from latent space if needed
            if self.args.latent:
                latents = 1 / 0.18215 * x
                image = self.vae.module.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
            else:
                image = (x.clamp(-1, 1) + 1) / 2
            
            # Convert to numpy
            image = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            image = (image * 255).astype(np.uint8)
            
            if image.shape[2] == 1:
                image = np.concatenate([image, image, image], axis=2)
            
            pil_image = Image.fromarray(image)
            
        return pil_image
    
    def generate_sentence(self, sentence, style_images, writer_id=0, 
                         num_inference_steps=50, gap=20):
        """Generate full sentence with proper word spacing"""
        words = sentence.strip().split()
        word_images_np = []
        
        print(f"\n📖 Generating sentence with {len(words)} words...")
        
        for i, word in enumerate(words):
            print(f"  [{i+1}/{len(words)}] {word}")
            # Generate word
            word_img_pil = self.generate(word, style_images, writer_id, num_inference_steps)
            word_img_np = np.array(word_img_pil)
            
            # Crop whitespace
            cropped_word = crop_whitespace(word_img_np)
            word_images_np.append(cropped_word)
        
        # Create gap image
        gap_img = np.ones((IMG_HEIGHT, gap, 3), dtype=np.uint8) * 255
        
        # Concatenate words with gaps
        sentence_parts = []
        for w_img in word_images_np:
            sentence_parts.append(w_img)
            sentence_parts.append(gap_img)
        
        # Remove last gap
        if sentence_parts:
            sentence_parts = sentence_parts[:-1]
        
        # Concatenate horizontally
        if sentence_parts:
            sentence_img = np.concatenate(sentence_parts, axis=1)
        else:
            sentence_img = np.ones((IMG_HEIGHT, 100, 3), dtype=np.uint8) * 255
        
        return Image.fromarray(sentence_img)

def set_seed(seed):
    """Set random seed for reproducibility"""
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
    """
    style_paths = []
    
    # Method 1: Direct image paths (HIGHEST PRIORITY)
    if args.style_image_paths:
        print("📌 Using style images from --style_image_paths")
        style_paths = args.style_image_paths
        
    # Method 2: Construct paths from writer ID and range
    elif args.style_id is not None and args.style_max is not None:
        wid = args.style_id
        print(f"📂 Constructing paths for Writer ID: {wid} (0 to {args.style_max})")
        for i in range(args.style_max + 1):
            formatted_path = os.path.join(args.data_root, f"writer_0{wid}", 
                                         f"w0{wid}_{i}.png")
            style_paths.append(formatted_path)
    
    # Method 3: Legacy explicit paths
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
    
    # Validate paths
    missing_paths = [p for p in style_paths if not os.path.exists(p)]
    if missing_paths:
        print(f"⚠️ Warning: {len(missing_paths)} style image(s) not found:")
        for p in missing_paths[:3]:
            print(f"    • {p}")
        if len(missing_paths) > 3:
            print(f"    ... and {len(missing_paths) - 3} more")
    
    return style_paths

def main():
    parser = argparse.ArgumentParser(
        description='Bengali Handwriting Generation Inference (Grapheme Tokenizer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With training pickle file (RECOMMENDED - ensures vocab match)
  python infer.py --model_path ./diffusionpen_bengali_model/models_new/ckpt.pt \\
      --pickle_path ./bengali_dataset.pickle \\
      --text "সে বই পড়ে" \\
      --style_image_paths style1.png style2.png
  
  # Method 1: Direct image paths
  python infer.py --model_path ./diffusionpen_bengali_model/models_new/ckpt.pt \\
      --text "আমার সোনার বাংলা" \\
      --style_image_paths style1.png style2.png style3.png
  
  # Method 2: Auto-construct paths from writer ID
  python infer.py --model_path model.pth --text "চন্দ্র" \\
      --style_id 11 --style_max 4 --data_root ./extracted_images
  
  # With vocabulary file
  python infer.py --model_path model.pth --text "সে বই পড়ে" \\
      --vocab_texts_file ./vocab_texts.txt \\
      --style_image_paths style1.png style2.png
        """
    )
    
    # Model paths
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--style_path', type=str, 
                       default='./style_models/mixed_bengali_mobilenetv2_100.pth',
                       help='Path to style extractor model')
    parser.add_argument('--stable_dif_path', type=str, 
                       default='runwayml/stable-diffusion-v1-5',
                       help='Path to stable diffusion model')
    parser.add_argument('--vocab_texts_file', type=str, default=None,
                       help='Path to file containing texts for vocabulary building')
    parser.add_argument('--pickle_path', type=str, default=None,
                       help='Path to training pickle file (to build vocab from same data)')
    
    # Style Arguments
    style_group = parser.add_argument_group('Style Image Input Methods (choose ONE)')
    style_group.add_argument('--style_image_paths', type=str, nargs='+',
                            help='[RECOMMENDED] Direct paths to style images')
    style_group.add_argument('--style_id', type=str, default=None,
                            help='Writer ID for file naming')
    style_group.add_argument('--style_max', type=int, default=None,
                            help='Highest index of style image')
    style_group.add_argument('--data_root', type=str, default='./extracted_images',
                            help='Root directory for extracted images')
    style_group.add_argument('--style_images', type=str, nargs='+',
                            help='[LEGACY] Explicit paths to style images')

    # Input/Output
    parser.add_argument('--text', type=str, default='সে গল্প করে',
                       help='Bengali text to generate')
    parser.add_argument('--output', type=str, default='output.png',
                       help='Output image path')
    parser.add_argument('--writer_id', type=int, default=0,
                       help='Writer ID for conditioning')

    # Generation params
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--mode', type=str, default='sentence',
                       choices=['word', 'sentence'],
                       help='Generation mode')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--gap', type=int, default=20,
                       help='Gap between words in pixels')

    # Model Config (must match training)
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Additional flags (must match UNet expectations)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--dataparallel', type=bool, default=False)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🇧🇩 Bengali Handwriting Generation - Grapheme Inference")
    print("=" * 60)
    
    if args.seed is not None:
        print(f"🔒 Setting fixed seed: {args.seed}")
        set_seed(args.seed)
    
    # Initialize inference engine
    inference = BengaliGraphemeInference(args)
    
    # Resolve and load style images
    style_paths = resolve_style_paths(args)
    print(f"\n🎨 Loading {len(style_paths)} style reference images...")
    style_tensor = inference.load_style_images(style_paths)
    print(f"✓ Style images loaded: {style_tensor.shape}")
    
    # Generate
    print(f"\n🚀 Starting generation...")
    if args.mode == 'word':
        output_image = inference.generate(
            args.text, style_tensor, 
            writer_id=args.writer_id, 
            num_inference_steps=args.num_steps
        )
    else:
        output_image = inference.generate_sentence(
            args.text, style_tensor,
            writer_id=args.writer_id,
            num_inference_steps=args.num_steps,
            gap=args.gap
        )
    
    # Save output
    output_dir = os.path.dirname(args.output)
    if output_dir and output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    output_image.save(args.output)
    
    print(f"\n✅ Generated image saved to: {args.output}")
    print("=" * 60)

if __name__ == "__main__":
    main()