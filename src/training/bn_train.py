import comet_ml # MUST be the first import
from comet_ml import Experiment
import os
import torch
import json
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import pickle
from diffusers import AutoencoderKL, DDIMScheduler
import random
from src.models.unet import UNetModel
from torchvision import transforms
from src.models.feature_extractor import ImageEncoder
from src.utils.auxilary_functions import *
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
import cv2
import string
# Start with standard English punctuation (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
punctuation = string.punctuation 

# Add Bengali specific punctuation
# '।' is the Dari (Bengali full stop)
punctuation += '।'
torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 
IMG_WIDTH = 256
IMG_HEIGHT = 64
experiment = None

# Bengali character classes - update this based on your dataset
c_classes = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঁংঃািীুূৃেৈোৌ্ৎ০১২৩৪৫৬৭৮৯!"#&\'()*+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}

### Label padding function ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index.get(i, 0) for i in labels]  # Use .get() to handle unknown characters
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": char_classes}
else:
    tokens = {"GO_TOKEN": char_classes, "END_TOKEN": char_classes + 1, "PAD_TOKEN": char_classes + 2}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)
print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens


# ===== PICKLE DATASET CLASS =====
class PickleDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading Bengali text images from pickle file"""
    
    def __init__(self, pickle_path, fixed_size=(64, 256), transforms=None, args=None, max_samples=None):
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.args = args
        
        print(f"Loading dataset from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Flatten the nested dictionary structure
        self.data = []
        self.writer_ids = []
        
        for writer_id, samples in data_dict['train'].items():
            for sample in samples:
                self.data.append(sample)
                self.writer_ids.append(writer_id)
        
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            self.writer_ids = self.writer_ids[:max_samples]
        
        print(f"Loaded {len(self.data)} samples from {len(set(self.writer_ids))} writers")
        
        # Create writer ID mapping
        unique_writers = sorted(set(self.writer_ids))
        self.writer_to_idx = {writer: idx for idx, writer in enumerate(unique_writers)}
        self.num_writers = len(unique_writers)
        
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_width, img_height = img.size
        target_height, target_width = self.fixed_size
        
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
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        writer_id = self.writer_ids[idx]
        
        img = sample['img']
        label = sample['label']
        
        img = self.preprocess_image(img)
        if self.transforms:
            img = self.transforms(img)
        
        writer_idx = self.writer_to_idx[writer_id]
        
        # Get 5 style images from same writer
        writer_samples = [i for i, wid in enumerate(self.writer_ids) if wid == writer_id]
        
        if len(writer_samples) >= 5:
            style_indices = np.random.choice(writer_samples, size=5, replace=False)
        else:
            style_indices = np.random.choice(writer_samples, size=5, replace=True)
        
        style_images = []
        for s_idx in style_indices:
            style_img = self.data[s_idx]['img']
            style_img = self.preprocess_image(style_img)
            if self.transforms:
                style_img = self.transforms(style_img)
            style_images.append(style_img)
        
        style_images = torch.stack(style_images)
        
        return img, label, torch.tensor(writer_idx, dtype=torch.long), style_images


def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])
    style_images = torch.stack([item[3] for item in batch])
    return images, labels, writer_ids, style_images


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)
def make_comparison_row(ref_tensors, gen_tensors, img_height=64, gap=10):
    """
    Stitches a single row: [Ref Word 1] [Ref 2] [Ref 3] [Ref 4] | [Gen Word 1] [Gen Word 2] ...
    """
    import numpy as np
    
    def tensor_to_np(t):
        # Denormalize [-1, 1] -> [0, 1]
        t = (t.detach().cpu() + 1) / 2.0
        t = t.clamp(0, 1)
        img = t.permute(1, 2, 0).numpy()
        # Handle Grayscale
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        return (img * 255).astype(np.uint8)

    # 1. Process Reference Images (Left Side)
    # We want exactly 4 images. If we have fewer, we pad.
    ref_imgs_np = [tensor_to_np(t) for t in ref_tensors]
    
    # Gap image (white)
    gap_img = np.ones((img_height, gap, 3), dtype=np.uint8) * 255
    
    # Stitch Left Side
    left_pieces = []
    for img in ref_imgs_np:
        left_pieces.append(img)
        left_pieces.append(gap_img)
    
    if left_pieces:
        left_side = np.concatenate(left_pieces[:-1], axis=1) # Remove last gap
    else:
        left_side = np.ones((img_height, 100, 3), dtype=np.uint8) * 255 # Fallback

    # 2. Process Generated Images (Right Side)
    gen_imgs_np = [tensor_to_np(t) for t in gen_tensors]
    
    right_pieces = []
    for img in gen_imgs_np:
        right_pieces.append(img)
        right_pieces.append(gap_img)
        
    if right_pieces:
        right_side = np.concatenate(right_pieces[:-1], axis=1)
    else:
        right_side = np.ones((img_height, 100, 3), dtype=np.uint8) * 255

    # 3. Create Separator (Red vertical line)
    separator = np.zeros((img_height, 5, 3), dtype=np.uint8)
    separator[:, :, 0] = 255 # Red
    
    # 4. Combine Row
    row = np.concatenate([left_side, separator, right_side], axis=1)
    return row

def save_comparison_grid(generated_images, gt_images, path, args):
    """Save side-by-side comparison of generated vs ground truth"""
    import torch
    
    # Ensure same number of images
    n = min(generated_images.size(0), gt_images.size(0))
    generated_images = generated_images[:n]
    gt_images = gt_images[:n]
    
    # Create alternating pattern: [GT1, Gen1, GT2, Gen2, ...]
    comparison = []
    for i in range(n):
        comparison.append(gt_images[i])
        comparison.append(generated_images[i])
    
    comparison = torch.stack(comparison)
    
    # Create grid with 2 columns (GT and Generated side by side)
    grid = torchvision.utils.make_grid(comparison, nrow=2, padding=2, normalize=False)
    
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    
    im.save(path)
    return im


def generate_trgan_style_page(generated_sentences_list, reference_sentences_list, sentence_text, args, gap=16):
    """
    Generate TRGAN-style page with 2 columns:
    - Left: Reference images (actual writer samples)
    - Right: Generated sentence (hardcoded text in writer's style)
    
    Args:
        generated_sentences_list: List of [N_words, C, H, W] tensors (generated for each writer)
        reference_sentences_list: List of [N_words, C, H, W] tensors (actual samples from each writer)
        sentence_text: The hardcoded sentence being generated
        args: Arguments object
        gap: Gap between images/lines
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Helper function to concatenate word images horizontally
    def concat_words_horizontal(word_tensors, gap):
        """Concatenate word images horizontally with gaps"""
        words_np = []
        for word_img in word_tensors:
            # Denormalize if needed
            if word_img.min() < 0:
                word_img = (word_img / 2 + 0.5).clamp(0, 1)
            
            word_img = word_img.cpu().numpy()
            if word_img.shape[0] == 3:  # RGB [C, H, W]
                word_img = np.transpose(word_img, (1, 2, 0))  # [H, W, C]
            else:  # Grayscale
                word_img = word_img[0]  # [H, W]
            
            words_np.append(word_img)
            # Add gap
            gap_img = np.ones((word_img.shape[0], gap)) if len(word_img.shape) == 2 else np.ones((word_img.shape[0], gap, word_img.shape[2]))
            words_np.append(gap_img)
        
        # Remove last gap and concatenate
        words_np = words_np[:-1]
        return np.concatenate(words_np, axis=1)
    
    # === BUILD LEFT COLUMN (REFERENCE IMAGES) ===
    left_column_lines = []
    for ref_sentence in reference_sentences_list:
        ref_line = concat_words_horizontal(ref_sentence, gap)
        left_column_lines.append(ref_line)
    
    # === BUILD RIGHT COLUMN (GENERATED SENTENCES) ===
    right_column_lines = []
    for gen_sentence in generated_sentences_list:
        gen_line = concat_words_horizontal(gen_sentence, gap)
        right_column_lines.append(gen_line)
    
    # === MAKE ALL LINES SAME WIDTH PER COLUMN ===
    max_left_width = max([line.shape[1] for line in left_column_lines])
    max_right_width = max([line.shape[1] for line in right_column_lines])
    
    def pad_line(line, target_width):
        if line.shape[1] < target_width:
            pad_width = target_width - line.shape[1]
            if len(line.shape) == 2:  # Grayscale
                line = np.concatenate([line, np.ones((line.shape[0], pad_width))], axis=1)
            else:  # RGB
                line = np.concatenate([line, np.ones((line.shape[0], pad_width, line.shape[2]))], axis=1)
        return line
    
    left_column_lines = [pad_line(line, max_left_width) for line in left_column_lines]
    right_column_lines = [pad_line(line, max_right_width) for line in right_column_lines]
    
    # === ADD GAPS BETWEEN LINES ===
    left_with_gaps = []
    right_with_gaps = []
    
    is_color = len(left_column_lines[0].shape) == 3
    gap_h = np.ones((gap, max_left_width, 3)) if is_color else np.ones((gap, max_left_width))
    
    for left_line, right_line in zip(left_column_lines, right_column_lines):
        left_with_gaps.append(left_line)
        left_with_gaps.append(gap_h if gap_h.shape[1] == left_line.shape[1] else np.ones((gap, left_line.shape[1], 3) if is_color else (gap, left_line.shape[1])))
        
        right_gap = np.ones((gap, right_line.shape[1], 3)) if is_color else np.ones((gap, right_line.shape[1]))
        right_with_gaps.append(right_line)
        right_with_gaps.append(right_gap)
    
    # Remove last gap
    left_with_gaps = left_with_gaps[:-1]
    right_with_gaps = right_with_gaps[:-1]
    
    # Stack vertically
    left_column = np.concatenate(left_with_gaps, axis=0)
    right_column = np.concatenate(right_with_gaps, axis=0)
    
    # === MAKE BOTH COLUMNS SAME HEIGHT ===
    max_height = max(left_column.shape[0], right_column.shape[0])
    
    def pad_height(column, target_height):
        if column.shape[0] < target_height:
            pad_height = target_height - column.shape[0]
            if len(column.shape) == 2:  # Grayscale
                column = np.concatenate([column, np.ones((pad_height, column.shape[1]))], axis=0)
            else:  # RGB
                column = np.concatenate([column, np.ones((pad_height, column.shape[1], column.shape[2]))], axis=0)
        return column
    
    left_column = pad_height(left_column, max_height)
    right_column = pad_height(right_column, max_height)
    
    # === ADD VERTICAL SEPARATOR ===
    separator_width = gap * 2
    separator = np.ones((max_height, separator_width, 3)) if is_color else np.ones((max_height, separator_width))
    
    # === COMBINE LEFT AND RIGHT ===
    page = np.concatenate([left_column, separator, right_column], axis=1)
    
    # === ADD HEADER WITH SENTENCE TEXT ===
    header_height = 50
    header = np.ones((header_height, page.shape[1], 3)) if is_color else np.ones((header_height, page.shape[1]))
    
    # Convert to PIL and add text
    page = (page * 255).astype(np.uint8)
    header = (header * 255).astype(np.uint8)
    
    if len(page.shape) == 2:  # Grayscale
        page = np.stack([page, page, page], axis=2)
        header = np.stack([header, header, header], axis=2)
    
    header_img = Image.fromarray(header, mode='RGB')
    page_img = Image.fromarray(page, mode='RGB')
    
    # Draw text on header
    draw = ImageDraw.Draw(header_img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf", 30)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
    
    draw.text((10, 10), f"Sentence: {sentence_text}", fill=(0, 0, 0), font=font)
    
    # Combine header and page
    final_page = np.concatenate([np.array(header_img), np.array(page_img)], axis=0)
    final_page_img = Image.fromarray(final_page, mode='RGB')
    
    return final_page_img


def save_reference_and_generated_comparison(generated_images, style_images, path, args, max_refs=3):
    """
    Save comparison grid showing reference images and generated output.
    Layout: [Ref1] [Ref2] [Ref3] [Generated] per row
    
    Args:
        generated_images: Generated handwriting [N, C, H, W]
        style_images: Reference images [N, 5, C, H, W]
        path: Save path
        args: Arguments object
        max_refs: Number of reference images to show per sample (1-5)
    """
    import torch
    import torchvision
    from PIL import Image
    
    n = generated_images.size(0)  # Number of samples
    max_refs = min(max_refs, 5)   # Limit to available references
    
    # Denormalize if needed (assuming images are in [-1, 1] or [0, 1] range)
    if generated_images.min() < 0:
        generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
    
    # Prepare all images for grid
    comparison = []
    
    for i in range(n):
        # Add reference images for this sample (show max_refs out of 5)
        for ref_idx in range(max_refs):
            ref_img = style_images[i, ref_idx]  # Get ref_idx-th reference
            if ref_img.min() < 0:
                ref_img = (ref_img / 2 + 0.5).clamp(0, 1)
            comparison.append(ref_img)
        
        # Add generated image for this sample
        comparison.append(generated_images[i])
    
    # Stack all images
    comparison = torch.stack(comparison)
    
    # Create grid: (max_refs + 1) columns per sample
    # Each row shows: [Ref1, Ref2, Ref3, ..., Generated]
    nrow = max_refs + 1
    grid = torchvision.utils.make_grid(comparison, nrow=nrow, padding=2, normalize=False)
    
    # Save image
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    
    im.save(path)
    return im


def load_weights_for_finetuning(model, checkpoint_path, new_num_classes):
    print(f"🔄 Loading weights from {checkpoint_path} for fine-tuning...")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = model.state_dict()
    filtered_state_dict = {}
    skipped_layers = []
    
    for k, v in checkpoint.items():
        # 1. Normalize Key: Strip 'module.' prefix to get the "base" name
        base_key = k.replace('module.', '')
        
        # 2. Find the corresponding key in the current model
        # The current model might use "module.base_key" or just "base_key"
        target_key = None
        if base_key in model_state:
            target_key = base_key
        elif f"module.{base_key}" in model_state:
            target_key = f"module.{base_key}"
            
        if target_key is None:
            # Key exists in checkpoint but not in current model (normal architecture change)
            continue
            
        # 3. Check for Embedding Layer Mismatch
        if 'label_emb.weight' in base_key:
            if v.shape != model_state[target_key].shape:
                print(f"⚠️  Skipping {base_key}: Shape mismatch (Old: {v.shape}, New: {model_state[target_key].shape})")
                skipped_layers.append(base_key)
                continue

        # 4. Add to filtered dict using the TARGET model's key format
        filtered_state_dict[target_key] = v

    # 5. Load
    model.load_state_dict(filtered_state_dict, strict=False)
    
    print(f"✅ Successfully loaded base weights! (Skipped {len(skipped_layers)} layers)")
def save_images(images, path, args, **kwargs):
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    im.save(path)
    return im


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


# ===== IMAGE LOGGING FUNCTIONS =====


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.img_size = img_size
        self.device = args.device
    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
            pass
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        
        with torch.no_grad():
            data = next(iter(test_loader))
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            images = images[:n]
            transcr = transcr[:n]
            s_id = s_id[:n]
            style_images = style_images[:n]

            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings).to(args.device)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
            
            reshaped_images = style_images.reshape(-1, 3, 64, 256)
            
            if style_extractor is not None:
                style_features = style_extractor(reshaped_images).to(args.device)
            else:
                style_features = None
        
            if args.latent == True:
                x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            noise_scheduler.set_timesteps(50)
            for time2 in noise_scheduler.timesteps:
                t_item = time2.item()
                t = (torch.ones(images.size(0)) * t_item).long().to(args.device)
                noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                prev_noisy_sample = noise_scheduler.step(noisy_residual, time2, x).prev_sample
                x = prev_noisy_sample
                    
        model.train()
        if args.latent == True:
            latents = 1 / 0.18215 * x
            image = vae.module.decode(latents).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)
        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

def create_page_from_words(word_tensors, max_width=1500, gap=16, img_height=64):
    """
    Mimics TRGAN's get_page_from_words.
    Takes a list of word tensors, arranges them into lines, and stacks them to form a page.
    """
    import numpy as np
    
    # 1. Convert tensors to numpy and denormalize
    words_np = []
    for w in word_tensors:
        # Move to CPU, detach
        w = w.detach().cpu()
        # Denormalize from [-1, 1] to [0, 1]
        w = (w + 1) / 2.0
        w = w.clamp(0, 1)
        
        # Convert to numpy [H, W, C]
        img = w.permute(1, 2, 0).numpy()
        
        # If grayscale, convert to RGB for consistency
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
            
        words_np.append(img)

    # 2. Arrange into lines
    lines = []
    current_line = []
    current_width = 0
    
    # Gap image
    gap_img = np.ones((img_height, gap, 3))
    
    for word_img in words_np:
        w_width = word_img.shape[1]
        
        if current_width + w_width + gap > max_width and len(current_line) > 0:
            # Finish current line
            lines.append(np.concatenate(current_line, axis=1))
            current_line = []
            current_width = 0
            
        current_line.append(word_img)
        current_line.append(gap_img)
        current_width += w_width + gap
        
    # Append the last line if exists
    if len(current_line) > 0:
        # Remove trailing gap
        current_line = current_line[:-1]
        lines.append(np.concatenate(current_line, axis=1))

    if not lines:
        return np.ones((img_height, max_width, 3))

    # 3. Pad lines to make them equal width
    real_max_width = max([l.shape[1] for l in lines])
    padded_lines = []
    vertical_gap = np.ones((16, real_max_width, 3)) # Vertical gap between lines
    
    for line in lines:
        pad_width = real_max_width - line.shape[1]
        if pad_width > 0:
            pad = np.ones((img_height, pad_width, 3))
            line = np.concatenate([line, pad], axis=1)
        padded_lines.append(line)
        padded_lines.append(vertical_gap)
        
    # 4. Stack vertically to form the page
    # Remove last vertical gap
    padded_lines = padded_lines[:-1]
    page = np.concatenate(padded_lines, axis=0)
    
    # Convert back to uint8 [0, 255]
    page = (page * 255).astype(np.uint8)
    
    return Image.fromarray(page)


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    
    # Cooling parameters
    import time
    import gc
    
    for epoch in range(args.epochs):
        print(f'\nEpoch: {epoch}/{args.epochs}')
        
        # === COOLING BREAK BEFORE HEAVY EPOCHS ===
        if epoch > 0 and epoch % args.cooling_interval == 0 and args.cooling_interval > 0:
            print(f"\n{'='*60}")
            print(f"🧊 COOLING BREAK at epoch {epoch}")
            print(f"{'='*60}")
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Check temperature before cooling
            temp_before = None
            if torch.cuda.is_available():
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        temp_before = int(result.stdout.strip())
                        print(f"🌡️  GPU Temperature before cooling: {temp_before}°C")
                except:
                    pass
            
            # Wait for cooldown
            print(f"⏳ Waiting {args.cooling_duration} seconds for GPU to cool down...")
            for i in range(args.cooling_duration, 0, -1):
                print(f"   Cooling: {i} seconds remaining...", end='\r')
                time.sleep(1)
            
            # Check temperature after cooling
            if temp_before:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        temp_after = int(result.stdout.strip())
                        temp_drop = temp_before - temp_after
                        print(f"\n🌡️  GPU Temperature after cooling: {temp_after}°C (dropped {temp_drop}°C)")
                except:
                    pass
            
            print(f"✅ Cooling complete! Resuming training...")
            print(f"{'='*60}\n")
        
        pbar = tqdm(loader)
        
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens) 
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            if style_extractor is not None:
                reshaped_images = style_images.reshape(-1, 3, 64, 256)
                style_features = style_extractor(reshaped_images)
            else:
                style_features = None

            if args.latent == True:
                with torch.no_grad():
                    images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                    images = images * 0.18215
                    latents = images
            
            noise = torch.randn(images.shape).to(images.device)
            num_train_timesteps = diffusion.noise_steps
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
            
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            x_t = noisy_images
            t = timesteps
            
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(x_t, timesteps=t, context=text_features, y=s_id, style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)
            
            if lr_scheduler is not None:
                lr_scheduler.step()
    
        # ===== IMAGE LOGGING EVERY 4 EPOCHS =====
        # ===== IMAGE LOGGING EVERY 4 EPOCHS =====
        # ===== IMAGE LOGGING EVERY 4 EPOCHS =====
        if epoch % 2 == 0:
            torch.cuda.empty_cache()
            
            # --- CONFIGURATION ---
            hardcoded_sentence = 'আমি বাংলায় গান গাই' 
            gen_words = hardcoded_sentence.split()
            n_gen_words = len(gen_words)
            target_num_writers = 5
            
            # --- PREPARE DATA ---
            # Fetch a batch. Note: test_loader usually isn't shuffled, so we might need
            # to check if we have enough unique writers.
            try:
                test_data = next(iter(test_loader))
            except StopIteration:
                test_data = next(iter(test_loader))
                
            batch_imgs = test_data[0]       # [B, C, H, W]
            batch_writer_ids = test_data[2] # [B]
            batch_styles = test_data[3]     # [B, 5, C, H, W]
            
            unique_writers = torch.unique(batch_writer_ids)
            # Ensure we don't crash if batch has fewer than 5 writers
            num_writers_to_log = min(target_num_writers, len(unique_writers))
            
            row_images = []
            
            print(f"Generating comparison rows for {num_writers_to_log} writers...")

            # --- LOOP THROUGH WRITERS ---
            for i in range(num_writers_to_log):
                writer_id = unique_writers[i]
                
                # --- LEFT COLUMN: GET 4 REAL WORDS ---
                # Find indices for this writer
                indices = (batch_writer_ids == writer_id).nonzero(as_tuple=True)[0]
                
                # Take up to 4 words. If fewer exist, take all of them.
                # Since we want "random" and batch is shuffled/random, taking first 4 is fine.
                selected_indices = indices[:4] 
                real_word_tensors = [batch_imgs[idx] for idx in selected_indices]
                
                # --- RIGHT COLUMN: GENERATE SENTENCE ---
                
                # 1. Prepare Style Input
                # We use the style bank from the first occurrence of this writer
                # Shape: [1, 5, C, H, W]
                style_ref = batch_styles[selected_indices[0]].unsqueeze(0) 
                
                # Repeat style for every word in the generated sentence
                # Shape: [N_gen, 5, C, H, W]
                style_input = style_ref.repeat(n_gen_words, 1, 1, 1, 1).to(args.device)
                writer_id_input = writer_id.repeat(n_gen_words).to(args.device)

                # 2. Prepare Text Features
                if args.model_name == 'wordstylist':
                    batch_word_embeddings = []
                    for word in gen_words:
                        word_embedding = label_padding(word, num_tokens)
                        word_embedding = torch.from_numpy(np.array(word_embedding, dtype="int64")).long()
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    text_features = tokenizer(gen_words, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)

                # 3. Extract Features
                if style_extractor:
                    reshaped_style = style_input.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(reshaped_style).to(args.device)
                else:
                    style_features = None

                # 4. Diffusion Generation
                ema_model.eval()
                noise_scheduler.set_timesteps(50)
                
                # Init noise
                if args.latent:
                    x = torch.randn((n_gen_words, 4, diffusion.img_size[0] // 8, diffusion.img_size[1] // 8)).to(args.device)
                else:
                    x = torch.randn((n_gen_words, 3, diffusion.img_size[0], diffusion.img_size[1])).to(args.device)

                with torch.no_grad():
                    for time2 in noise_scheduler.timesteps:
                        t = (torch.ones(n_gen_words) * time2.item()).long().to(args.device)
                        noisy_residual = ema_model(x, t, text_features, writer_id_input, 
                                                original_images=style_input, 
                                                mix_rate=args.mix_rate, 
                                                style_extractor=style_features)
                        x = noise_scheduler.step(noisy_residual, time2, x).prev_sample
                
                # Decode
                if args.latent:
                    latents = 1 / 0.18215 * x
                    image = vae.module.decode(latents).sample
                else:
                    image = x
                
                generated_word_tensors = [image[k] for k in range(n_gen_words)]

                # --- STITCH ROW ---
                # Pass to helper function
                row_img_np = make_comparison_row(real_word_tensors, generated_word_tensors)
                row_images.append(row_img_np)
            
            # --- STACK ROWS AND SAVE ---
            if row_images:
                
                from PIL import Image
                
                # Make all rows same width (padding right side with white)
                max_w = max([r.shape[1] for r in row_images])
                padded_rows = []
                
                # Vertical gap between rows
                v_gap = np.ones((20, max_w, 3), dtype=np.uint8) * 255
                
                for row in row_images:
                    if row.shape[1] < max_w:
                        pad = np.ones((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8) * 255
                        row = np.concatenate([row, pad], axis=1)
                    padded_rows.append(row)
                    padded_rows.append(v_gap) # Add gap after each row
                
                # Remove last gap
                final_grid = np.concatenate(padded_rows[:-1], axis=0)
                final_pil = Image.fromarray(final_grid)
                
                save_path = os.path.join(args.save_path, 'images', f"{epoch}_comparison.jpg")
                final_pil.save(save_path)
                
                if args.wandb_log:
                    experiment.log_image(final_pil, name=f"epoch_{epoch}_comparison", step=epoch)

            # Save Checkpoints
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))
        # if epoch % 4 == 0:
        #     torch.cuda.empty_cache() 
            
        #     # === HARDCODED BENGALI SENTENCE ===
        #     hardcoded_sentence = 'আমি বাংলায় গান গাই'  # "I sing in Bengali"
        #     words = hardcoded_sentence.split()
        #     n_words = len(words)
            
        #     # Number of different writers to show
        #     num_writers = min(8, args.batch_size)
            
        #     all_generated_sentences = []
        #     all_reference_sentences = []
            
        #     # Generate for multiple writers
        #     for writer_idx in range(num_writers):
        #         with torch.no_grad():
        #             # Get unique writer from test set
        #             test_data = next(iter(test_loader))
                    
        #             # Get reference images - actual samples from this writer
        #             style_images_writer = test_data[3][writer_idx:writer_idx+1].repeat(n_words, 1, 1, 1, 1).to(args.device)  # [N_words, 5, C, H, W]
        #             writer_id_writer = test_data[2][writer_idx:writer_idx+1].repeat(n_words).to(args.device)
                    
        #             # Get actual reference sentence from this writer (left column)
        #             # Use first n_words samples from this writer
        #             reference_images_writer = test_data[0][writer_idx:writer_idx+1].repeat(n_words, 1, 1, 1).to(args.device)  # [N_words, C, H, W]
                
        #         # === GENERATE HARDCODED SENTENCE IN THIS WRITER'S STYLE (RIGHT COLUMN) ===
        #         # Prepare text features for hardcoded sentence words
        #         if args.model_name == 'wordstylist':
        #             batch_word_embeddings = []
        #             for word in words:
        #                 word_embedding = label_padding(word, num_tokens)
        #                 word_embedding = np.array(word_embedding, dtype="int64")
        #                 word_embedding = torch.from_numpy(word_embedding).long() 
        #                 batch_word_embeddings.append(word_embedding)
        #             text_features_words = torch.stack(batch_word_embeddings).to(args.device)
        #         else:
        #             text_features_words = tokenizer(words, padding="max_length", truncation=True, return_tensors="pt", max_length=200).to(args.device)
                
        #         # Extract style features
        #         reshaped_images = style_images_writer.reshape(-1, 3, 64, 256)
        #         if style_extractor is not None:
        #             style_features = style_extractor(reshaped_images).to(args.device)
        #         else:
        #             style_features = None
                
        #         # Sample using EMA model
        #         if args.latent == True:
        #             x = torch.randn((n_words, 4, diffusion.img_size[0] // 8, diffusion.img_size[1] // 8)).to(args.device)
        #         else:
        #             x = torch.randn((n_words, 3, diffusion.img_size[0], diffusion.img_size[1])).to(args.device)
                
        #         ema_model.eval()
        #         noise_scheduler.set_timesteps(50)
                
        #         with torch.no_grad():
        #             for time in noise_scheduler.timesteps:
        #                 t_item = time.item()
        #                 t = (torch.ones(n_words) * t_item).long().to(args.device)
        #                 noisy_residual = ema_model(x, t, text_features_words, writer_id_writer, 
        #                                           original_images=style_images_writer, 
        #                                           mix_rate=args.mix_rate, 
        #                                           style_extractor=style_features)
        #                 prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
        #                 x = prev_noisy_sample
                
        #         # Decode if using latent mode
        #         if args.latent == True:
        #             latents = 1 / 0.18215 * x
        #             image = vae.module.decode(latents).sample
        #             image = (image / 2 + 0.5).clamp(0, 1)
        #             image = image.cpu().permute(0, 2, 3, 1).numpy()
        #             image = torch.from_numpy(image)
        #             generated_words = image.permute(0, 3, 1, 2)
        #         else:
        #             generated_words = (x.clamp(-1, 1) + 1) / 2
        #             generated_words = (generated_words * 255).type(torch.uint8)
                
        #         # Store generated sentence (right column)
        #         all_generated_sentences.append(generated_words)
                
        #         # Store reference images (left column)
        #         all_reference_sentences.append(reference_images_writer)
            
        #     epoch_n = epoch
            
        #     # === GENERATE TRGAN-STYLE PAGE ===
        #     # Left column: Reference images (actual samples from different writers)
        #     # Right column: Generated hardcoded sentence in each writer's style
        #     page_img = generate_trgan_style_page(
        #         all_generated_sentences,  # Right column: generated hardcoded sentence
        #         all_reference_sentences,   # Left column: actual writer samples
        #         hardcoded_sentence,
        #         args,
        #         gap=16
        #     )
        #     page_img.save(os.path.join(args.save_path, 'images', f"{epoch_n}_trgan_page.jpg"))
            
        #     # === LOG TO COMET ML ===
        #     if args.wandb_log == True:
        #         experiment.log_image(
        #             page_img, 
        #             name=f"trgan_style_page_{epoch}", 
        #             step=epoch,
        #             image_format="png" 
        #         )
                
        #         experiment.log_text(
        #             f"Hardcoded Sentence: {hardcoded_sentence}\nGenerated for {num_writers} different writers",
        #             step=epoch
        #         )
            
        #     # Save model checkpoints
        #     torch.save(model.state_dict(), os.path.join(args.save_path, "models2", "ckpt.pt"))
        #     torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models2", "ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models2", "optim.pt"))   

def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    #rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)

def save_single_images(images, path, args):
    """
    Saves a batch of images as a single file. 
    If batch_size > 1, it creates a grid.
    """
    import os
    import torch
    import torchvision
    import numpy as np
    from PIL import Image

    # 1. Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2. Denormalize: Check if images are in [-1, 1] range (common in diffusion)
    # If min value is negative, shift to [0, 1]
    if images.min() < 0:
        images = (images + 1) / 2.0
    
    # Clamp to ensure valid range [0, 1]
    images = images.clamp(0, 1)

    # 3. Handle Batching
    if images.dim() == 4 and images.size(0) > 1:
        # Create a grid layout if there are multiple images
        # nrow controls how many images per row
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=False)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
    elif images.dim() == 4 and images.size(0) == 1:
        # Single batch item
        ndarr = images[0].permute(1, 2, 0).cpu().numpy()
    else:
        # Assuming [C, H, W] single tensor
        ndarr = images.permute(1, 2, 0).cpu().numpy()

    # 4. Handle Grayscale vs RGB
    if ndarr.shape[2] == 1:
        # Replicate channels to make it RGB (compatible with most viewers)
        ndarr = np.concatenate([ndarr, ndarr, ndarr], axis=2)

    # 5. Convert to PIL and Save
    im = Image.fromarray((ndarr * 255).astype(np.uint8))
    im.save(path)
    # Optional: Print confirmation
    # prdef save_single_images(images, path, args):
    """
    Saves a batch of images as a single file. 
    If batch_size > 1, it creates a grid.
    """
    

    # 1. Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 2. Denormalize: Check if images are in [-1, 1] range (common in diffusion)
    # If min value is negative, shift to [0, 1]
    if images.min() < 0:
        images = (images + 1) / 2.0
    
    # Clamp to ensure valid range [0, 1]
    images = images.clamp(0, 1)

    # 3. Handle Batching
    if images.dim() == 4 and images.size(0) > 1:
        # Create a grid layout if there are multiple images
        # nrow controls how many images per row
        grid = torchvision.utils.make_grid(images, nrow=4, padding=2, normalize=False)
        ndarr = grid.permute(1, 2, 0).cpu().numpy()
    elif images.dim() == 4 and images.size(0) == 1:
        # Single batch item
        ndarr = images[0].permute(1, 2, 0).cpu().numpy()
    else:
        # Assuming [C, H, W] single tensor
        ndarr = images.permute(1, 2, 0).cpu().numpy()

    # 4. Handle Grayscale vs RGB
    if ndarr.shape[2] == 1:
        # Replicate channels to make it RGB (compatible with most viewers)
        ndarr = np.concatenate([ndarr, ndarr, ndarr], axis=2)

    # 5. Convert to PIL and Save
    im = Image.fromarray((ndarr * 255).astype(np.uint8))
    im.save(path)
    # Optional: Print confirmation
    # print(f"Saved sample to {path}")int(f"Saved sample to {path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    
    # NEW: Path to pickle dataset
    parser.add_argument('--pickle_path', type=str, default='./bengali_dataset.pickle', 
                        help='Path to the pickle file containing the dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to load (None = all)')
    
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_bengali_model') 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--style_path', type=str, default='./style_models/bengali_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='bengali')
    parser.add_argument('--cooling_interval', type=int, default=3,
                        help='Cool down GPU every N epochs (set 0 to disable)')
    parser.add_argument('--cooling_duration', type=int, default=10,
                        help='Cooling break duration in seconds')
    parser.add_argument('--power_limit', type=int, default=None,
                        help='GPU power limit in watts (e.g., 350 for RTX 4090)')
    
    args = parser.parse_args()
   
    # Ensure device is properly set
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print(f"WARNING: CUDA not available, switching from {args.device} to CPU")
        args.device = 'cpu'
    
    print(f'Using device: {args.device}')
    print('torch version', torch.__version__)
    
    # === APPLY POWER LIMIT IF SPECIFIED ===
    if args.power_limit is not None and torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '-pl', str(args.power_limit)],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"✅ GPU power limit set to {args.power_limit}W")
            else:
                print(f"⚠️ Could not set power limit: {result.stderr}")
                print("   Try running as Administrator/sudo or set manually")
        except Exception as e:
            print(f"⚠️ Could not set power limit: {e}")
            print("   Please run: nvidia-smi -pl {args.power_limit} manually")
    
    # === GPU TEMPERATURE MONITORING ===
    def get_gpu_temp():
        """Get current GPU temperature"""
        if not torch.cuda.is_available():
            return None
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
        except:
            pass
        return None
    
    # Check initial GPU temperature
    initial_temp = get_gpu_temp()
    if initial_temp:
        print(f"📊 Current GPU temperature: {initial_temp}°C")
        if initial_temp > 80:
            print(f"⚠️ WARNING: GPU is already hot ({initial_temp}°C)!")
            print("   Consider cooling down before starting training")
    
    print(f"\n{'='*60}")
    print(f"🔧 Training Configuration:")
    print(f"{'='*60}")
    print(f"  Cooling interval: Every {args.cooling_interval} epochs")
    print(f"  Cooling duration: {args.cooling_duration} seconds")
    if args.power_limit:
        print(f"  Power limit: {args.power_limit}W")
    print(f"{'='*60}\n")
    
    if args.wandb_log == True:
        global experiment
        experiment = Experiment(
            api_key="6xk1Nmcm6P2OmkiUlYSqe4IqV",
            project_name="diffusionpen-bengali",
            workspace="rabib-jahin",
            auto_output_logging="simple"
        )
    
    setup_logging(args)

    transform = transforms.Compose([
        transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load pickle dataset
    print('Loading Bengali pickle dataset...')
    full_dataset = PickleDataset(
        pickle_path=args.pickle_path,
        fixed_size=(64, 256),
        transforms=transform,
        args=args,
        max_samples=args.max_samples
    )
    
    style_classes = full_dataset.num_writers
    print(f'Number of unique writers (style classes): {style_classes}')
    
    # Split into train and test
    test_size = args.batch_size
    rest = len(full_dataset) - test_size
    train_data, test_data = random_split(
        full_dataset, 
        [rest, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'Train samples: {len(train_data)}, Test samples: {len(test_data)}')
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    if args.dataparallel == True and torch.cuda.is_available():
        device_ids = [3, 4]
        print('using dataparallel with device:', device_ids)
    else:
        if torch.cuda.is_available():
            idx = int(''.join(filter(str.isdigit, args.device))) if any(c.isdigit() for c in args.device) else 0
        else:
            idx = 0
        device_ids = [idx]

    # Load tokenizer and text encoder
    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None
    
    # Create UNet model
    if args.unet == 'unet_latent':
        unet = UNetModel(
            image_size=args.img_size, 
            in_channels=args.channels, 
            model_channels=args.emb_dim, 
            out_channels=args.channels, 
            num_res_blocks=args.num_res_blocks, 
            attention_resolutions=(1, 1), 
            channel_mult=(1, 1), 
            num_heads=args.num_heads, 
            num_classes=style_classes, 
            context_dim=args.emb_dim, 
            vocab_size=vocab_size, 
            text_encoder=text_encoder, 
            args=args
        )
    
    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    if args.load_check == True:
        load_weights_for_finetuning(unet, "./models/models/ckpt.pt",0)
        # unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        # optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        # ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
    
    if args.latent == True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # Load style extractor
    # CRITICAL: num_classes should be 0 for feature extraction only
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    PATH = args.style_path 
    
    if os.path.exists(PATH):
        print(f"\n=== Loading Style Encoder ===")
        print(f"Path: {PATH}")
        state_dict = torch.load(PATH, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        
        # Filter and match keys
        matched_keys = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        unmatched_keys = [k for k in state_dict.keys() if k not in matched_keys]
        
        print(f"Matched keys: {len(matched_keys)} / {len(state_dict)}")
        print(f"Total model keys: {len(model_dict)}")
        if unmatched_keys:
            print(f"⚠️ Unmatched keys (first 5): {unmatched_keys[:5]}")
        
        model_dict.update(matched_keys)
        feature_extractor.load_state_dict(model_dict, strict=False)
        print(f"✓ Successfully loaded style extractor")
        print("=============================\n")
    else:
        print(f"⚠️ WARNING: Style model path {PATH} not found!")
        print(f"Training without pretrained style weights.\n")

    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    if args.train_mode == 'train':
        train(
            diffusion, 
            unet, 
            ema, 
            ema_model, 
            vae, 
            optimizer, 
            mse_loss, 
            train_loader, 
            test_loader, 
            style_classes, 
            feature_extractor, 
            vocab_size, 
            ddim, 
            transform, 
            args, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            lr_scheduler=lr_scheduler
        )
        
    elif args.train_mode == 'sampling':
        
        print('Sampling started....')
        
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        unet.eval()
        
        ema = EMA(0.995)
        ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        ema_model.eval()
        
        if args.sampling_mode == 'single_sampling':
            x_text = ['text', 'word']
            for x_text in x_text:
                print('Word:', x_text)
                s = random.randint(0, 339) #index for style class
                
                print('style', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)  
                save_single_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)

        
        elif args.sampling_mode == 'paragraph':
            print('Sampling paragraph')
            #make the code to generate lines
            lines = 'এই গবেষণায়, আমরা শৈলীর ভিন্নতার ওপর আলোকপাত করেছি। আমরা লেখার শৈলী নিয়ন্ত্রণ করার জন্য একটি অভিনব পদ্ধতি উপস্থাপন করছি। আমাদের পদ্ধতিটি বিভিন্ন লেখার শৈলী অনুকরণ করতে সক্ষম।'
            # lines = 'In this work , we focus on style variation . We present a nove method to control the style of the text . Our method is able to mimic various writing styles .'
            # fakes= []
            gap = np.ones((64, 16))
            max_line_width = 900
            total_char_count = 0
            avg_char_width = 0
            current_line_width = 0
            longest_word_length = max(len(word) for word in lines.strip().split(' '))
            #print('longest_word_length', longest_word_length)
            #s = random.randint(0, 339)#.long().to(args.device)
            #s = random.randint(0, 161)#.long().to(args.device)
            s = 12 #25 #129 #201
            for word in lines.strip().split(' '):
                print('Word:', word)
                print('Style:', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)  
                #print('ema_sampled_images', ema_sampled_images.shape)
                image = ema_sampled_images.squeeze(0)
                
                im = torchvision.transforms.ToPILImage()(image)
                #reshape to height 32
                im = im.convert("L")
                #save im
                
                #if len(word) < 4:
                    
                im = crop_whitespace_width(im)
                
                im = Image.fromarray(im)
                if len(word) == longest_word_length:
                    max_word_length_width = im.width
                    print('max_word_length_width', max_word_length_width)
                #im.save(f'./_REBUTTAL/{word}.png')
                # Calculate aspect ratio
                aspect_ratio = im.width / im.height
                
                im = np.array(im)
                #im = np.array(resized_img)
                
                fakes.append(im)
            
            # Calculate the scaling factor based on the longest word
            #find the average character width of the max length word
            
            avg_char_width = max_word_length_width / longest_word_length
            print('avg_char_width', avg_char_width)
            #scaling_factor = avg_char_width / (32 * aspect_ratio)  # Aspect ratio of an average character

            # Scale and pad each word
            scaled_padded_words = []
            max_height = 64  # Defined max height for all images
            
            for word, img in zip(lines.strip().split(' '), fakes):
                
                img_pil = Image.fromarray(img)
                as_ratio = img_pil.width / img_pil.height
                #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
                scaled_width = int(avg_char_width * len(word))
                
                scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
                print(f'Word {word} - scaled_img {scaled_img.size}')
                # Padding
                #if word is in punctuation:
                if word in punctuation:
                    #rescale to height 10
                    w_punc = scaled_img.width
                    h_punc = scaled_img.height
                    as_ratio_punct = w_punc / h_punc
                    if word == '.':
                        scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
                    else:
                        scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
                    #pad on top and leave the image in the bottom
                    padding_bottom = 10
                    padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
                      # No padding at the bottom

                    # Apply padding
                    padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
                else:
                    if scaled_img.height < max_height:
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    else:
                        #resize to max height while maintaining aspect ratio
                        #ar = scaled_img.width / scaled_img.height
                        
                        scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                        
                    #padded_img = np.array(scaled_img)
                #print('padded_img', padded_img.shape)
                scaled_padded_words.append(padded_img)

            # Create a gap array (white space)
            height = 64  # Fixed height for all images
            gap = np.ones((height, 16), dtype=np.uint8) * 255  # White gap

            # Concatenate images with gaps
            sentence_img = gap  # Start with a gap
            lines = [] 
            line_img = gap
            # Concatenate images with gaps
            '''
            sentence_img = gap  # Start with a gap
            for img in scaled_padded_words:
                #print('img', img.shape)
                sentence_img = np.concatenate((sentence_img, img, gap), axis=1)
            '''
            
            for img in scaled_padded_words:
                img_width = img.shape[1] + gap.shape[1]

                if current_line_width + img_width < max_line_width:
                    # Add the image to the current line
                    if line_img.shape[0] == 0:
                        line_img = np.ones((height, 0), dtype=np.uint8) * 255  # Start a new line
                    line_img = np.concatenate((line_img, img, gap), axis=1)
                    current_line_width += img_width #+ gap.shape[1]
                    #print('current_line_width if', current_line_width)
                    # Check if adding this image exceeds the max line width
                else:
                    # Pad the current line with white space to max_line_width
                    remaining_width = max_line_width - current_line_width
                    line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                    lines.append(line_img)

                    # Start a new line with the current word
                    line_img = np.concatenate((gap, img, gap), axis=1)
                    current_line_width = img_width #+ 2 * gap.shape[1]
                    #print('current_line_width else', current_line_width)
            # Add the last line to the lines list
            if current_line_width > 0:
                # Pad the last line to max_line_width
                remaining_width = max_line_width - current_line_width
                line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                lines.append(line_img)
                
            # # Concatenate all lines to form a paragraph, pad them if necessary
            # max_height = max([line.shape[0] for line in lines])
            # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
            # for line in lines:
            #     if line.shape[0] < max_height:
            #         padding = (max_height - line.shape[0]) // 2
            #         line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)
                
            #     #print the shapes
            #     print('line shape', line.shape)
            #print('paragraph shape', paragraph_img.shape)
            paragraph_img = np.concatenate((lines), axis=0)

                
            paragraph_image = Image.fromarray(paragraph_img)
            paragraph_image = paragraph_image.convert("L")    
            
            paragraph_image.save(f'paragraph_style_{s}.png')


if __name__ == "__main__":
    main()