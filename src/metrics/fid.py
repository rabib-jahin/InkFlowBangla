import torch
import numpy as np
from PIL import Image
import argparse
import os
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel
import torch.nn as nn
from torch.nn import DataParallel
import pickle
from tqdm import tqdm
import random

# Import your custom modules
from src.models.unet import UNetModel
from src.models.feature_extractor import ImageEncoder

# Configuration
OUTPUT_MAX_LEN = 95
IMG_WIDTH = 256
IMG_HEIGHT = 64

# Bengali character classes
c_classes = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঁংঃািীুূৃেৈোৌ্ৎ০১২৩৪৫৬৭৮৯!"#&\'()*+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()
tokens = {"PAD_TOKEN": char_classes}
num_tokens = len(tokens.keys())
vocab_size = char_classes + num_tokens


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
        from PIL import ImageOps
        img = ImageOps.pad(img, size=(target_width, target_height), color="white")
    elif img_width > target_width:
        left = (img_width - target_width) // 2
        img = img.crop((left, 0, left + target_width, target_height))
    
    return img


class GeneratorForFID:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        
        print(f"🔧 Initializing generator on {self.device}")
        
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
            print(f"✓ Loaded style extractor")
        
        self.style_extractor = DataParallel(self.style_extractor)
        self.style_extractor = self.style_extractor.to(self.device)
        self.style_extractor.requires_grad_(False)
        self.style_extractor.eval()
        
        # Load UNet
        print("🧠 Loading UNet model...")
        
        num_classes = 1
        if os.path.exists(args.model_path):
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'module.label_emb.weight' in checkpoint:
                num_classes = checkpoint['module.label_emb.weight'].shape[0]
            elif 'label_emb.weight' in checkpoint:
                num_classes = checkpoint['label_emb.weight'].shape[0]
        
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
        
        if os.path.exists(args.model_path):
            self.model.load_state_dict(checkpoint)
            print(f"✓ Loaded model")
        
        self.model.eval()
        
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.stable_dif_path, 
            subfolder="scheduler"
        )
        
        print("✅ Initialization complete!\n")
    
    def load_style_images(self, style_images_pil):
        """Load style images from PIL images"""
        style_tensors = []
        for img in style_images_pil:
            img = preprocess_image(img, fixed_size=(64, 256))
            img_tensor = self.transform(img)
            style_tensors.append(img_tensor)
        
        while len(style_tensors) < 5:
            style_tensors.append(style_tensors[-1])
        style_tensors = style_tensors[:5]
        
        style_tensor = torch.stack(style_tensors).unsqueeze(0)
        return style_tensor.to(self.device)
    
    def generate(self, text, style_images, writer_id=0):
        """Generate a single image"""
        with torch.no_grad():
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
                x = torch.randn((1, 3, IMGf_HEIGHT, IMG_WIDTH)).to(self.device)
            
            self.noise_scheduler.set_timesteps(self.args.num_steps)
            writer_id_tensor = torch.tensor([writer_id]).long().to(self.device)
            
            for time_step in self.noise_scheduler.timesteps:
                t = torch.ones(1).long().to(self.device) * time_step.item()
                noisy_residual = self.model(
                    x, t, text_features, writer_id_tensor,
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
            
            return Image.fromarray(image)


def generate_images_matching_dataset(args):
    """
    Generate images that match the real dataset distribution:
    - Same texts
    - Same writer styles (using style reference images from dataset)
    - Same proportions
    """
    print("=" * 60)
    print("🎨 Generating Images for FID Calculation")
    print("   Strategy: Match Real Dataset Distribution")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset from {args.pickle_path}...")
    with open(args.pickle_path, 'rb') as f:
        data_dict = pickle.load(f)
    
    # Organize data by writer
    writer_data = {}
    for writer_id, samples in data_dict['train'].items():
        writer_data[writer_id] = samples
    
    print(f"✓ Found {len(writer_data)} writers")
    print(f"✓ Total samples: {sum(len(samples) for samples in writer_data.values())}")
    
    # Initialize generator
    generator = GeneratorForFID(args)
    
    # Create output folder
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Keep track of writer ID mapping (original → model index)
    unique_writers = sorted(writer_data.keys())
    writer_to_idx = {writer: idx for idx, writer in enumerate(unique_writers)}
    
    print(f"\n🎯 Generation Strategy:")
    print(f"   • For each writer in dataset:")
    print(f"     1. Use their handwriting as style reference")
    print(f"     2. Generate the SAME texts they wrote")
    print(f"   • Total to generate: {args.num_samples} images")
    print()
    
    # Determine which samples to generate
    all_samples = []
    for writer_id, samples in writer_data.items():
        for sample in samples:
            all_samples.append({
                'text': sample['label'],
                'writer_id': writer_id,
                'style_images': samples  # All samples from this writer for style
            })
    
    # Limit to num_samples
    if len(all_samples) > args.num_samples:
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.num_samples)
    
    print(f"📝 Generating {len(all_samples)} images...")
    
    # Generate images
    generated_count = 0
    
    for idx, sample_info in enumerate(tqdm(all_samples, desc="Generating")):
        text = sample_info['text']
        writer_id_original = sample_info['writer_id']
        writer_id_model = writer_to_idx[writer_id_original]
        
        # Select random style reference images from this writer
        style_samples = sample_info['style_images']
        num_style_refs = min(5, len(style_samples))
        style_ref_samples = random.sample(style_samples, num_style_refs)
        
        # Extract style images
        style_images_pil = [s['img'] for s in style_ref_samples]
        
        # Load and prepare style
        style_tensor = generator.load_style_images(style_images_pil)
        
        # Generate image
        try:
            generated_image = generator.generate(text, style_tensor, writer_id_model)
            
            # Save image
            output_path = os.path.join(args.output_folder, f"gen_{idx:06d}_w{writer_id_original}.png")
            generated_image.save(output_path)
            
            generated_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error generating image {idx}: {e}")
            continue
    
    print(f"\n✅ Generation complete!")
    print(f"   • Successfully generated: {generated_count}/{len(all_samples)} images")
    print(f"   • Saved to: {args.output_folder}")
    print(f"\n💡 Next step: Run FID calculation")
    print(f"   python calculate_fid.py \\")
    print(f"       --real_images_pickle {args.pickle_path} \\")
    print(f"       --generated_images_folder {args.output_folder}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Images for FID Calculation (Matching Real Distribution)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_for_fid.py \\
      --model_path ./models/ckpt.pt \\
      --pickle_path ./bengali_dataset.pickle \\
      --output_folder ./generated_for_fid \\
      --num_samples 1000 \\
      --num_steps 50
        """
    )
    
    # Paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--style_path', type=str, 
                       default='./style_models/bengali_style_diffusionpen.pth',
                       help='Path to style extractor model')
    parser.add_argument('--pickle_path', type=str, required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('--output_folder', type=str, default='./generated_for_fid',
                       help='Output folder for generated images')
    
    # Generation settings
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Model config
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    generate_images_matching_dataset(args)


if __name__ == "__main__":
    main()