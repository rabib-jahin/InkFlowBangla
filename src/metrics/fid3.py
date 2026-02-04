import torch
import numpy as np
from PIL import Image, ImageOps
import argparse
import os
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler
import torch.nn as nn
from torch.nn import DataParallel
import pickle
from tqdm import tqdm
import random

# Fix OMP Error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        
        # Load Grapheme Tokenizer
        print("📝 Loading Bengali Grapheme Tokenizer...")
        self.tokenizer = BengaliGraphemeTokenizer(output_max_len=OUTPUT_MAX_LEN)
        
        # Build vocabulary from dataset
        print("🔨 Building vocabulary from dataset...")
        with open(args.pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        all_texts = []
        for writer_id, samples in data_dict['train'].items():
            for sample in samples:
                all_texts.append(sample['label'])
        
        self.tokenizer.build_vocab(all_texts)
        vocab_size = self.tokenizer.get_vocab_size()
        print(f"✓ Vocabulary built: {vocab_size} graphemes")
        
        self.text_encoder = None  # No text encoder needed for grapheme tokenizer
        
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
    
    def prepare_style_batch(self, style_images_pil_list):
        """
        Input: List of List of PIL images. Shape: [Batch_Size, 5]
        Output: Tensor of shape [Batch_Size, 5, 3, 64, 256]
        """
        batch_tensors = []
        for style_imgs in style_images_pil_list:
            processed_imgs = []
            for img in style_imgs:
                img = preprocess_image(img, fixed_size=(64, 256))
                img_tensor = self.transform(img)
                processed_imgs.append(img_tensor)
            
            # Ensure exactly 5 images
            while len(processed_imgs) < 5:
                processed_imgs.append(processed_imgs[-1])
            processed_imgs = processed_imgs[:5]
            
            batch_tensors.append(torch.stack(processed_imgs))
            
        return torch.stack(batch_tensors).to(self.device)

    def generate_batch(self, texts, style_images_batch, writer_ids):
        """
        Generate a batch of images using Grapheme Tokenizer
        texts: List of strings
        style_images_batch: Tensor [B, 5, 3, 64, 256]
        writer_ids: Tensor [B]
        """
        batch_size = len(texts)
        
        with torch.no_grad():
            # 1. Text Encoding using Grapheme Tokenizer
            batch_indices = []
            for text in texts:
                encoded = self.tokenizer.encode(text)
                batch_indices.append(torch.tensor(encoded))
            
            text_features = torch.stack(batch_indices).long().to(self.device)
            
            # 2. Style Encoding
            reshaped_images = style_images_batch.view(-1, 3, 64, 256)
            style_features = self.style_extractor(reshaped_images)
            
            # 3. Initialize Latents
            if self.args.latent:
                x = torch.randn((batch_size, 4, IMG_HEIGHT // 8, IMG_WIDTH // 8)).to(self.device)
            else:
                x = torch.randn((batch_size, 3, IMG_HEIGHT, IMG_WIDTH)).to(self.device)
            
            self.noise_scheduler.set_timesteps(self.args.num_steps)
            
            # 4. Denoising Loop
            for time_step in self.noise_scheduler.timesteps:
                t = torch.ones(batch_size).long().to(self.device) * time_step.item()
                
                noisy_residual = self.model(
                    x, t, text_features, writer_ids,
                    style_extractor=style_features
                )
                x = self.noise_scheduler.step(noisy_residual, time_step, x).prev_sample
            
            # 5. Decode Latents
            if self.args.latent:
                latents = 1 / 0.18215 * x
                image = self.vae.module.decode(latents).sample
                image = (image / 2 + 0.5).clamp(0, 1)
            else:
                image = (x.clamp(-1, 1) + 1) / 2
            
            # Convert to numpy batch
            images = image.permute(0, 2, 3, 1).cpu().numpy()
            images = (images * 255).astype(np.uint8)
            
            if images.shape[3] == 1:
                images = np.concatenate([images, images, images], axis=3)
            
            return [Image.fromarray(img) for img in images]


def generate_images_matching_dataset(args):
    print("=" * 60)
    print(f"🎨 Generating Images with Grapheme Tokenizer")
    print(f"   Batch Size: {args.batch_size}")
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
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Writer mapping
    unique_writers = sorted(writer_data.keys())
    writer_to_idx = {writer: idx for idx, writer in enumerate(unique_writers)}
    
    # Prepare samples
    all_samples = []
    for writer_id, samples in writer_data.items():
        for sample in samples:
            all_samples.append({
                'text': sample['label'],
                'writer_id': writer_id,
                'style_images': samples
            })
    
    # Limit samples
    if len(all_samples) > args.num_samples:
        random.seed(args.seed)
        all_samples = random.sample(all_samples, args.num_samples)
    
    print(f"\n📝 Total images to generate: {len(all_samples)}")
    print(f"🎯 Starting batch generation...\n")
    
    # Batch generation loop
    generated_count = 0
    
    for i in range(0, len(all_samples), args.batch_size):
        batch_samples = all_samples[i : i + args.batch_size]
        
        # Prepare batch data
        batch_texts = []
        batch_style_imgs = []
        batch_writer_ids = []
        original_writer_ids = []
        
        for sample in batch_samples:
            batch_texts.append(sample['text'])
            
            wid_orig = sample['writer_id']
            original_writer_ids.append(wid_orig)
            batch_writer_ids.append(writer_to_idx[wid_orig])
            
            style_set = sample['style_images']
            selected_styles = random.sample(style_set, min(5, len(style_set)))
            batch_style_imgs.append([s['img'] for s in selected_styles])
        
        try:
            writer_ids_tensor = torch.tensor(batch_writer_ids).long().to(args.device)
            style_tensor_batch = generator.prepare_style_batch(batch_style_imgs)
            
            # Generate batch
            generated_images = generator.generate_batch(
                batch_texts, 
                style_tensor_batch, 
                writer_ids_tensor
            )
            
            # Save batch
            for j, img in enumerate(generated_images):
                global_idx = i + j
                orig_wid = original_writer_ids[j]
                
                output_path = os.path.join(args.output_folder, f"gen_{global_idx:06d}_w{orig_wid}.png")
                img.save(output_path)
                generated_count += 1
            
            # Memory cleanup
            if i % (args.batch_size * 2) == 0:
                torch.cuda.empty_cache()

            print(f"\rProgress: {generated_count}/{len(all_samples)}", end="")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n⚠️ OOM Error at batch {i}. Reduce batch size! Skipping batch.")
                torch.cuda.empty_cache()
            else:
                print(f"\n❌ Error: {e}")
        except Exception as e:
            print(f"\n⚠️ Error generating batch {i}: {e}")

    print(f"\n\n✅ Generation complete!")
    print(f"   Successfully generated: {generated_count}/{len(all_samples)} images")
    print(f"   Saved to: {args.output_folder}")
    print(f"\n💡 Next step: Run FID calculation")
    print(f"   python calculate_fid.py \\")
    print(f"       --real_images_pickle {args.pickle_path} \\")
    print(f"       --generated_images_folder {args.output_folder}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Images for FID with Grapheme Tokenizer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Core Paths
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--style_path', type=str, 
                       default='./style_models/bengali_style_diffusionpen.pth',
                       help='Path to style extractor')
    parser.add_argument('--pickle_path', type=str, required=True,
                       help='Path to dataset pickle file')
    parser.add_argument('--output_folder', type=str, default='./generated_for_fid_grapheme_15000',
                       help='Output folder for generated images')
    
    # Generation Settings
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of denoising steps')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for generation (reduce if OOM)')
    
    # Model Config
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--unet', type=str, default='unet_latent')
    
    # Additional Args
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--color', type=bool, default=True)
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    generate_images_matching_dataset(args)


if __name__ == "__main__":
    main()