import comet_ml
import torch.nn.functional as F
from src.models.tokenizer import BengaliGraphemeTokenizer
from comet_ml import Experiment
import os
import torch
import json
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, random_split
import torch.cuda.amp as amp  # Import AMP for Mixed Precision
import torchvision
from tqdm import tqdm
from torch import optim
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
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

punctuation = string.punctuation 
punctuation += '।'
torch.cuda.empty_cache()

OUTPUT_MAX_LEN = 95 
IMG_WIDTH = 256
IMG_HEIGHT = 64
experiment = None

start_epoch=12

# Bengali character classes
c_classes = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঁংঃািীুূৃেৈোৌ্ৎ০১২৩৪৫৬৭৮৯!"#&\'()*+,-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}

### Label padding function ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index.get(i, 0) for i in labels]
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

class MatraConsistencyLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        # Horizontal Edge Detection Filter
        self.sobel = torch.tensor([[[[-1, -2, -1],
                                     [ 0,  0,  0],
                                     [ 1,  2,  1]]]], dtype=torch.float32).to(device)
        self.sobel.requires_grad = False

    def forward(self, pred_img, target_img):
        # ছবির উপরের ৪০% অংশ (যেখানে মাত্রা থাকে) কেটে নেওয়া
        h_limit = int(pred_img.shape[2] * 0.4) 
        pred_top = pred_img[:, :, :h_limit, :].mean(dim=1, keepdim=True)
        target_top = target_img[:, :, :h_limit, :].mean(dim=1, keepdim=True)
        
        # এজ ডিটেকশন এবং লস
        pred_edge = F.conv2d(pred_top, self.sobel, padding=1)
        target_edge = F.conv2d(target_top, self.sobel, padding=1)
        return F.mse_loss(pred_edge, target_edge)
# ===== PICKLE DATASET CLASS =====
class PickleDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading Bengali text images from pickle file"""
    
    def __init__(self, pickle_path, fixed_size=(64, 256), transforms=None, args=None, max_samples=None, load_style_images=True):
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.args = args
        self.load_style_images = load_style_images
        
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
        
        # Create writer ID mapping - ENSURE INTEGER INDICES
        unique_writers = sorted(set(self.writer_ids))
        self.writer_to_idx = {writer: idx for idx, writer in enumerate(unique_writers)}
        self.idx_to_writer = {idx: writer for idx, writer in enumerate(unique_writers)}
        self.num_writers = len(unique_writers)
        
        print(f"📋 Writer ID mapping created:")
        print(f"    Sample original writer_ids: {unique_writers[:5]}")
        print(f"    Sample mapped indices: {[self.writer_to_idx[w] for w in unique_writers[:5]]}")
        print(f"    Mapping type check: {type(self.writer_to_idx[unique_writers[0]])}")
        
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
        
        if self.load_style_images:
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
        else:
            # When using cache, don't load style images
            return img, label, torch.tensor(writer_idx, dtype=torch.long), None


# ===== STYLE FEATURE CACHING =====
def precompute_writer_style_features(dataset, style_extractor, args, samples_per_writer=10):
    """
    Pre-compute style features for each writer (not each sample).
    For each writer, we store features from multiple reference samples.
    Uses writer_idx (0 to N-1) as INTEGER keys.
    """
    print("\n" + "="*60)
    print("🔧 Pre-computing writer style features...")
    print("="*60)
    
    style_extractor.eval()
    writer_style_cache = {}  # {writer_idx (int): style_features}
    
    # Group samples by writer_idx (not writer_id)
    writer_samples = {}
    for idx in range(len(dataset)):
        writer_id = dataset.writer_ids[idx]
        writer_idx = dataset.writer_to_idx[writer_id]  # Use mapped index (0 to N-1)
        
        # Ensure writer_idx is an integer
        if isinstance(writer_idx, str):
            raise ValueError(f"writer_idx should be int, got string: {writer_idx}")
        
        if writer_idx not in writer_samples:
            writer_samples[writer_idx] = []
        writer_samples[writer_idx].append(idx)
    
    print(f"📊 Found {len(writer_samples)} unique writers")
    print(f"📦 Caching {samples_per_writer} samples per writer")
    print(f"📋 Writer mapping example:")
    print(f"    Original writer_ids (first 5): {list(set(dataset.writer_ids))[:5]}")
    print(f"    Mapped writer_idx (first 5): {sorted(list(writer_samples.keys()))[:5]}")
    
    with torch.no_grad():
        for writer_idx, sample_indices in tqdm(writer_samples.items(), desc="Processing writers"):
            # Get N style samples from this writer
            num_samples = min(samples_per_writer, len(sample_indices))
            selected_indices = np.random.choice(sample_indices, size=num_samples, replace=False)
            
            style_features_list = []
            for idx in selected_indices:
                sample = dataset.data[idx]
                img = sample['img']
                img = dataset.preprocess_image(img)
                if dataset.transforms:
                    img = dataset.transforms(img)
                img = img.unsqueeze(0).to(args.device)  # [1, C, H, W]
                
                # Extract features
                feat = style_extractor(img)  # [1, feat_dim]
                style_features_list.append(feat.cpu())
            
            # Stack all style features for this writer - use INTEGER writer_idx as key
            writer_style_cache[int(writer_idx)] = torch.stack(style_features_list).squeeze(1)  # [N, feat_dim]
    
    print(f"✅ Cached style features for {len(writer_style_cache)} writers")
    print(f"💾 Cache size: ~{sum(v.numel() * 4 / (1024**2) for v in writer_style_cache.values()):.2f} MB")
    print(f"📋 Cache keys (first 10): {sorted(list(writer_style_cache.keys()))[:10]}")
    print(f"📋 Cache keys type: {type(list(writer_style_cache.keys())[0])}")
    print("="*60 + "\n")
    
    return writer_style_cache


def collate_fn_with_cache(batch, style_cache, num_style_images=5):
    """
    Custom collate function that uses cached style features.
    Expects writer_idx (0 to N-1) from dataset, not original writer_id.
    """
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])  # These are writer_idx values
    
    # Instead of loading style_images, we'll use cached features
    batch_style_features = []
    
    # Debug: Check first writer_idx
    if len(writer_ids) > 0:
        first_idx = writer_ids[0].item()
        if isinstance(first_idx, str):
            raise TypeError(f"Writer indices should be integers, got string: {first_idx}")
    
    for writer_idx_tensor in writer_ids:
        writer_idx = int(writer_idx_tensor.item())  # Ensure it's an integer
        
        # Get cached features for this writer using writer_idx
        if writer_idx not in style_cache:
            cache_keys = sorted(list(style_cache.keys()))
            raise KeyError(
                f"Writer index {writer_idx} not found in cache.\n"
                f"Cache has {len(style_cache)} writers with indices: {cache_keys[:20]}...\n"
                f"Cache key type: {type(cache_keys[0]) if cache_keys else 'empty'}\n"
                f"Looking for type: {type(writer_idx)}\n"
                f"This usually means the cache was created with a different dataset split."
            )
        
        cached_features = style_cache[writer_idx]  # [N_cached, feat_dim]
        
        # Randomly sample num_style_images features
        if cached_features.size(0) >= num_style_images:
            indices = np.random.choice(cached_features.size(0), size=num_style_images, replace=False)
        else:
            indices = np.random.choice(cached_features.size(0), size=num_style_images, replace=True)
        
        sampled_features = cached_features[indices]  # [num_style_images, feat_dim]
        batch_style_features.append(sampled_features)
    
    batch_style_features = torch.stack(batch_style_features)  # [B, num_style_images, feat_dim]
    
    return images, labels, writer_ids, batch_style_features


def collate_fn(batch):
    """Original collate function (fallback when not using cache)"""
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])
    style_images = torch.stack([item[3] for item in batch])
    return images, labels, writer_ids, style_images


# ===== HELPER FUNCTIONS =====
def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models_new'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images_new'), exist_ok=True)


def make_comparison_row(ref_tensors, gen_tensors, img_height=64, gap=10):
    """
    Stitches a single row: [Ref Word 1] [Ref 2] [Ref 3] [Ref 4] | [Gen Word 1] [Gen Word 2] ...
    """
    import numpy as np
    
    def tensor_to_np(t):
        t = (t.detach().cpu() + 1) / 2.0
        t = t.clamp(0, 1)
        img = t.permute(1, 2, 0).numpy()
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        return (img * 255).astype(np.uint8)

    ref_imgs_np = [tensor_to_np(t) for t in ref_tensors]
    gap_img = np.ones((img_height, gap, 3), dtype=np.uint8) * 255
    
    left_pieces = []
    for img in ref_imgs_np:
        left_pieces.append(img)
        left_pieces.append(gap_img)
    
    if left_pieces:
        left_side = np.concatenate(left_pieces[:-1], axis=1)
    else:
        left_side = np.ones((img_height, 100, 3), dtype=np.uint8) * 255

    gen_imgs_np = [tensor_to_np(t) for t in gen_tensors]
    
    right_pieces = []
    for img in gen_imgs_np:
        right_pieces.append(img)
        right_pieces.append(gap_img)
        
    if right_pieces:
        right_side = np.concatenate(right_pieces[:-1], axis=1)
    else:
        right_side = np.ones((img_height, 100, 3), dtype=np.uint8) * 255

    separator = np.zeros((img_height, 5, 3), dtype=np.uint8)
    separator[:, :, 0] = 255
    
    row = np.concatenate([left_side, separator, right_side], axis=1)
    return row


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

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))


# ===== TRAINING FUNCTION WITH CACHING =====
import torch.cuda.amp as amp  # Import AMP for Mixed Precision

# ===== OPTIMIZED TRAINING FUNCTION =====
def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, 
          num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, 
          tokenizer=None, text_encoder=None, lr_scheduler=None, use_cached_features=False):
    
    model.train()
    loss_meter = AvgMeter()
    print('\n🚀 Training started with Optimized Memory Management....')
    
  

    if use_cached_features:
        print("✅ Using cached style features (fast mode)")
    else:
        print("⚠️ Computing style features on-the-fly (slow mode)")
    
    import time
    import gc
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'📅 Epoch: {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        if epoch > 0 and epoch % args.cooling_interval == 0 and args.cooling_interval > 0:
            print(f"\n{'='*60}")
            print(f"🧊 COOLING BREAK at epoch {epoch}")
            print(f"{'='*60}")
            # torch.cuda.empty_cache()
            # gc.collect()
            
            
            
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
            
            print(f"⏳ Waiting {args.cooling_duration} seconds for GPU to cool down...")
            for i in range(args.cooling_duration, 0, -1):
                print(f"   Cooling: {i} seconds remaining...", end='\r')
                time.sleep(1)
            
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
        
        
        # === COOLING BREAK (Optimized) ===
        # Only run cooling if strictly necessary. 
        # With AMP and del, the GPU shouldn't overheat as fast.

        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        epoch_start_time = time.time()
        
        for i, data in enumerate(pbar):
            # 1. Move Data to Device
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_data = data[3]
            
            # 2. Text Features (No Gradients needed here usually)
            with torch.no_grad():
                if args.model_name == 'wordstylist':
                    batch_word_embeddings = []
                    for trans in transcr:
                        word_embedding = label_padding(trans, num_tokens) 
                        word_embedding = np.array(word_embedding, dtype="int64")
                        word_embedding = torch.from_numpy(word_embedding).long() 
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    batch_indices = []
                    for text in transcr:
                        encoded = tokenizer.encode(text)
                        batch_indices.append(torch.tensor(encoded))
                    text_features = torch.stack(batch_indices).long().to(args.device)

                # Style Features
                if use_cached_features:
                    style_features = style_data.to(args.device)
                    B, N, feat_dim = style_features.shape
                    style_features = style_features.reshape(B * N, feat_dim)
                else:
                    if style_extractor is not None:
                        reshaped_images = style_data.reshape(-1, 3, 64, 256).to(args.device)
                        style_features = style_extractor(reshaped_images)
                    else:
                        style_features = None

                # VAE Encoding (Latent)
                if args.latent:
                    if hasattr(vae, 'module'):
    # DataParallel wrapped
                        images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                    else:
    # Single GPU
                        images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                    # images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                    images = images * 0.18215
            
            # 3. Noise Generation
            noise = torch.randn(images.shape).to(images.device)
            num_train_timesteps = diffusion.noise_steps
            timesteps = torch.randint(0, num_train_timesteps, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Classifier-free guidance
            labels = s_id
            if np.random.random() < 0.1:
                labels = None
            
            optimizer.zero_grad(set_to_none=True) # Better than zero_grad() for memory

            # === MIXED PRECISION TRAINING (AMP) ===
            # This runs the forward pass in float16, saving massive VRAM
            # with amp.autocast():
            predicted_noise = model(noisy_images, timesteps=timesteps, context=text_features, y=labels, 
                                    style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise)

        # Backward Pass with Scaler
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
            ema.step_ema(ema_model, model)

        # Update Meters
            count = images.size(0)
            loss_val = loss.item() # Get float value
            loss_meter.update(loss_val, count)
            pbar.set_postfix(MSE=loss_meter.avg)

            # === AGGRESSIVE MEMORY CLEANUP ===
            # Delete tensors that are no longer needed for this iteration
           
            
            # Periodically clear CUDA cache (Not every step, causes slowdown. Every 50 steps is good)
            if i % 10 == 0:
                torch.cuda.empty_cache()

        # End of Epoch Stats
        epoch_time = time.time() - epoch_start_time
        print(f"\n⏱️  Epoch time: {epoch_time:.2f}s")
        print(f"📊 Average Loss: {loss_meter.avg:.4f}")
        
        if lr_scheduler is not None:
            lr_scheduler.step()

        # === IMAGE LOGGING & VALIDATION ===
        if epoch % 2 == 0:
            print(f"\n📸 Generating comparison images...")
            
            # Clear memory specifically before generation to avoid OOM spikes
            # torch.cuda.empty_cache()
            
            hardcoded_sentence = 'সে বই পড়ে' 
            gen_words = hardcoded_sentence.split()
            n_gen_words = len(gen_words)
            target_num_writers = 5
            
            try:
                # Use no_grad to ensure no graph is built
                with torch.no_grad():
                    try:
                        test_data = next(iter(test_loader))
                    except StopIteration:
                        test_data = next(iter(test_loader))
                    
                    batch_imgs = test_data[0]
                    batch_writer_ids = test_data[2]
                    
                    # Logic to handle style features (same as before)
                    # ... [Your existing logic for style prep] ...
                    if use_cached_features:
                         batch_styles = test_data[3]
                    else:
                         batch_styles = test_data[3]
                    
                    unique_writers = torch.unique(batch_writer_ids)
                    num_writers_to_log = min(target_num_writers, len(unique_writers))
                    row_images = []

                    for idx in range(num_writers_to_log):
                        # ... [Your logic for selecting indices] ...
                        writer_id = unique_writers[idx]
                        indices = (batch_writer_ids == writer_id).nonzero(as_tuple=True)[0]
                        selected_indices = indices[:4]
                        real_word_tensors = [batch_imgs[sidx] for sidx in selected_indices]

                        # Style Input Setup
                        if use_cached_features:
                            style_ref = batch_styles[selected_indices[0]].unsqueeze(0)
                            style_input_feat = style_ref.repeat(n_gen_words, 1, 1).to(args.device)
                            B, N, feat_dim = style_input_feat.shape
                            style_features_gen = style_input_feat.reshape(B * N, feat_dim)
                        else:
                            style_ref = batch_styles[selected_indices[0]].unsqueeze(0)
                            style_input = style_ref.repeat(n_gen_words, 1, 1, 1, 1).to(args.device)
                            reshaped_style = style_input.reshape(-1, 3, 64, 256)
                            style_features_gen = style_extractor(reshaped_style).to(args.device)

                        writer_id_input = writer_id.repeat(n_gen_words).to(args.device)

                        # Text Features Gen
                        batch_gen_indices = []
                        for word in gen_words:
                            encoded = tokenizer.encode(word) 
                            batch_gen_indices.append(torch.tensor(encoded))
                        text_features_gen = torch.stack(batch_gen_indices).long().to(args.device)

                        # Sampling
                        ema_model.eval()
                        noise_scheduler.set_timesteps(50)
                        
                        if args.latent:
                             x = torch.randn((n_gen_words, 4, diffusion.img_size[0] // 8, 
                                            diffusion.img_size[1] // 8)).to(args.device)
                        else:
                             x = torch.randn((n_gen_words, 3, diffusion.img_size[0], 
                                            diffusion.img_size[1])).to(args.device)
                        
                        for time2 in noise_scheduler.timesteps:
                            t = (torch.ones(n_gen_words) * time2.item()).long().to(args.device)
                            # NO AMP HERE (Usually inference is fine in fp32, or use autocast if needed)
                            noisy_residual = ema_model(x, t, text_features_gen, writer_id_input, 
                                                     style_extractor=style_features_gen)
                            x = noise_scheduler.step(noisy_residual, time2, x).prev_sample
                        
                        if args.latent:
                            latents = 1 / 0.18215 * x
                            if hasattr(vae, 'module'):
                                image = vae.module.decode(latents).sample
                            else:
                                image = vae.decode(latents).sample
                            # image = vae.module.decode(latents).sample
                        else:
                            image = x
                        
                        generated_word_tensors = [image[k] for k in range(n_gen_words)]
                        row_img_np = make_comparison_row(real_word_tensors, generated_word_tensors)
                        row_images.append(row_img_np)
                        
                        # CLEANUP INTERMEDIATE TENSORS
                        # del x, noisy_residual, image, style_features_gen, text_features_gen
            
            except Exception as e:
                print(f"Error generating images: {e}")
            
            # Save Image Logic
            if row_images:
                # ... [Your image saving logic] ...
                from PIL import Image
                max_w = max([r.shape[1] for r in row_images])
                padded_rows = []
                v_gap = np.ones((20, max_w, 3), dtype=np.uint8) * 255
                for row in row_images:
                    if row.shape[1] < max_w:
                        pad = np.ones((row.shape[0], max_w - row.shape[1], 3), dtype=np.uint8) * 255
                        row = np.concatenate([row, pad], axis=1)
                    padded_rows.append(row)
                    padded_rows.append(v_gap)
                
                final_grid = np.concatenate(padded_rows[:-1], axis=0)
                final_pil = Image.fromarray(final_grid)
                save_path = os.path.join(args.save_path, 'images_new', f"{epoch}_comparison.jpg")
                final_pil.save(save_path)
                print(f"✅ Saved comparison image to {save_path}")

            # === Checkpoint Saving ===
            print(f"💾 Saving checkpoints...")
            torch.save(model.state_dict(), os.path.join(args.save_path, "models_new", f"ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models_new", f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models_new", f"optim.pt"))
            print(f"✅ Checkpoints saved")
            
            # Cleanup after generation
           

    print("\n✅ Training complete!")


# ===== MAIN FUNCTION =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    
    parser.add_argument('--pickle_path', type=str, default='./bengali_dataset.pickle')
    parser.add_argument('--max_samples', type=int, default=None)
    
    # === NEW: CACHING ARGUMENTS ===
    parser.add_argument('--cache_style_features', type=bool, default=True,
                        help='Enable writer-level style feature caching')
    parser.add_argument('--style_cache_path', type=str, default='./style_cache/writer_styles.pt',
                        help='Path to save/load cached style features')
    parser.add_argument('--num_style_samples', type=int, default=5,
                        help='Number of samples to cache per writer')
    parser.add_argument('--num_style_images', type=int, default=5,
                        help='Number of style features to use per batch')
    
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_bengali_model') 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=False)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/bengali_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--dataset', type=str, default='bengali')
    parser.add_argument('--cooling_interval', type=int, default=3)
    parser.add_argument('--cooling_duration', type=int, default=10)
    parser.add_argument('--power_limit', type=int, default=None)
    
    args = parser.parse_args()
    args.latent=True
    # args.latent= False
    if args.latent:
        args.channels = 4  # Latent Space (Standard VAE Output)
        print("✅ Mode: Latent Space (Channels: 4)")
    else:
        args.channels = 3  # Pixel Space (RGB Image)
        print("✅ Mode: Pixel Space (Channels: 3)")
   
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print(f"WARNING: CUDA not available, switching from {args.device} to CPU")
        args.device = 'cpu'
    
    print(f'\n{"="*60}')
    print(f'🚀 DiffusionPen Bengali Training')
    print(f'{"="*60}')
    print(f'Device: {args.device}')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Style caching: {"✅ Enabled" if args.cache_style_features else "❌ Disabled"}')
    print(f'{"="*60}\n')
    
    # Power limit
    if args.power_limit is not None and torch.cuda.is_available():
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-pl', str(args.power_limit)],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ GPU power limit set to {args.power_limit}W")
        except Exception as e:
            print(f"⚠️ Could not set power limit: {e}")
    
    # Comet ML
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
    print('📦 Loading Bengali pickle dataset...')
    full_dataset = PickleDataset(
        pickle_path=args.pickle_path,
        fixed_size=(64, 256),
        transforms=transform,
        args=args,
        max_samples=args.max_samples,
        load_style_images=not args.cache_style_features  # Don't load if caching
    )
    
    style_classes = full_dataset.num_writers
    print(f'✅ Number of unique writers (style classes): {style_classes}')
    
    # Device setup
    if args.dataparallel == True and torch.cuda.is_available():
        device_ids = [0]
        print('Using DataParallel with devices:', device_ids)
    else:
        if torch.cuda.is_available():
            idx = int(''.join(filter(str.isdigit, args.device))) if any(c.isdigit() for c in args.device) else 0
        else:
            idx = 0
        device_ids = [idx]

    # Load tokenizer
    if args.model_name == 'diffusionpen':
        print('\n🔤 Setting up Grapheme Tokenizer...')
    
        # ১. টোকেনাইজার ইনিশিওলাইজ করুন
        tokenizer = BengaliGraphemeTokenizer(output_max_len=95) # আপনার OUTPUT_MAX_LEN অনুযায়ী
        
        # ২. ডেটাসেট থেকে সব টেক্সট বের করুন (ভোকাবুলারি বানানোর জন্য)
        # আপনার full_dataset অবজেক্টটি অলরেডি লোড করা আছে
        all_texts = [sample['label'] for sample in full_dataset.data]
        
        # ৩. ভোকাবুলারি বিল্ড করুন
        tokenizer.build_vocab(all_texts)
        
        # ৪. নতুন ভোকাব সাইজ সেট করুন
        vocab_size = tokenizer.get_vocab_size()
        print(f"🔢 New Vocab Size: {vocab_size}")

        # Text Encoder এখন None থাকবে কারণ আমরা সরাসরি ইনডেক্স পাস করব
        text_encoder = None
        # tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        # text_encoder = CanineModel.from_pretrained("google/canine-c")
        # text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        # text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None
    
    # === STYLE EXTRACTOR AND CACHING ===
    print('\n📥 Loading style extractor...')
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, 
                                     pretrained=True, trainable=True)
    PATH = args.style_path 
    
    if os.path.exists(PATH):
        print(f"Loading from: {PATH}")
        state_dict = torch.load(PATH, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        matched_keys = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched_keys)
        feature_extractor.load_state_dict(model_dict, strict=False)
        print(f"✅ Style extractor loaded ({len(matched_keys)}/{len(state_dict)} keys)")
    else:
        print(f"⚠️ Style model not found at {PATH}, using random initialization")

    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    # feature_extractor.eval()
    feature_extractor.requires_grad_(True)
    feature_extractor.train()
    # === PRE-COMPUTE OR LOAD STYLE CACHE ===
    writer_style_cache = None
    cache_path = args.style_cache_path
    
    if args.cache_style_features:
        if not os.path.exists(cache_path):
            print(f"\n🔄 Cache not found, creating new cache...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            writer_style_cache = precompute_writer_style_features(
                full_dataset, 
                feature_extractor, 
                args,
                samples_per_writer=args.num_style_samples
            )
            torch.save(writer_style_cache, cache_path)
            print(f"✅ Saved writer style cache to {cache_path}")
        else:
            print(f"📂 Loading cached writer styles from {cache_path}")
            writer_style_cache = torch.load(cache_path, map_location='cpu')
            print(f"✅ Loaded cache for {len(writer_style_cache)} writers")
    
    # Split into train and test
    test_size = args.batch_size
    rest = len(full_dataset) - test_size
    train_data, test_data = random_split(
        full_dataset, 
        [rest, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'📊 Train samples: {len(train_data)}, Test samples: {len(test_data)}')
    
    # === CREATE DATA LOADERS ===
    if writer_style_cache is not None:
        print(f"🚀 Using cached style features (fast mode)")
        collate_fn_cached = lambda batch: collate_fn_with_cache(
            batch, writer_style_cache, num_style_images=args.num_style_images
        )
        
        train_loader = DataLoader(
            train_data, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=collate_fn_cached
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=False,
            collate_fn=collate_fn_cached
        )
    else:
        print(f"⚠️ Computing features on-the-fly (slow mode)")
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=False, collate_fn=collate_fn)
    
    # === CREATE UNET ===
    print('\n🏗️ Building UNet model...')
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
    print(f"✅ UNet created")
    start_epoch=12
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
   
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    if args.load_check == True:
        print('\n📥 Loading checkpoints...')
        unet.load_state_dict(torch.load(f'{args.save_path}/models_new/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models_new/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models_new/ema_ckpt.pt'))
        print('✅ Loaded models and optimizer')
        start_epoch=12
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', 0.0001)
    
            start_epoch = 12# Set the epoch to resume from
            print(f'✅ Loaded models and optimizer, resuming from epoch {start_epoch}')
    # VAE
    lr_scheduler = None


    if args.latent == True:
        print('📥 Loading VAE...')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        # vae = vae.to(args.device) 
        vae = vae.to(args.device)
        vae.requires_grad_(False)
        vae.eval()
        print('✅ VAE loaded')
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    # === START TRAINING ===
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
            feature_extractor if not writer_style_cache else None,  # Don't pass if using cache
            vocab_size, 
            ddim, 
            transform, 
            args, 
            tokenizer=tokenizer, 
            text_encoder=text_encoder, 
            lr_scheduler=lr_scheduler,
            use_cached_features=writer_style_cache is not None
        )
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()