import comet_ml
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
import gc
import time

# ==========================================
# 1. EXPANDED VOCABULARY FOR MULTILINGUAL
# ==========================================
punctuation = string.punctuation + '।?!.,:;-\'\"' 
# Bengali chars + English Letters + Digits + Punctuation + Space
c_classes = 'অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঁংঃািীুূৃেৈোৌ্ৎ০১২৩৪৫৬৭৮৯' + \
            string.ascii_letters + string.digits + punctuation + ' '

cdict = {c: i for i, c in enumerate(c_classes)}
icdict = {i: c for i, c in enumerate(c_classes)}

torch.cuda.empty_cache()

OUTPUT_MAX_LEN = 95 
IMG_WIDTH = 256
IMG_HEIGHT = 64
experiment = None

# Set this to 0 if starting fresh, or the epoch you want to resume from
start_epoch = 19

def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    index2letter = {v: k for k, v in letter2index.items()}
    return len(labels), letter2index, index2letter

char_classes, letter2index, index2letter = labelDictionary()

# Token setup
tokens = {"PAD_TOKEN": char_classes}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)
print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens

### Safe Label Padding Function ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = []
    for i in labels:
        # SAFETY CHECK: If character is not in our list, print warning or skip
        if i in letter2index:
            ll.append(letter2index[i])
        else:
            # You might want to uncomment this to see what characters are missing
            # print(f"⚠️ Warning: Character '{i}' not in vocabulary, skipping.")
            pass
    
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)
    return ll

# ==========================================
# 2. UPDATED DATASET CLASS (Multilingual)
# ==========================================
class PickleDataset(torch.utils.data.Dataset):
    """Custom Dataset for loading Multilingual text images from multiple pickle files"""
    
    def __init__(self, pickle_paths, fixed_size=(64, 256), transforms=None, args=None, max_samples=None, load_style_images=True):
        self.fixed_size = fixed_size
        self.transforms = transforms
        self.args = args
        self.load_style_images = load_style_images
        
        self.data = []
        self.writer_ids = []

        # Handle single path string input by converting to list
        if isinstance(pickle_paths, str):
            pickle_paths = [pickle_paths]

        print(f"📦 Loading datasets from: {pickle_paths}")

        # Loop through all provided pickle files
        for path in pickle_paths:
            path = path.strip() # Remove spaces if any
            if not os.path.exists(path):
                print(f"❌ Error: File not found {path}")
                continue
                
            print(f"   -> Loading {path}...")
            try:
                with open(path, 'rb') as f:
                    data_dict = pickle.load(f)
                
                # Prefix to avoid writer ID collisions (e.g., 'ben_writer1', 'eng_writer1')
                prefix = os.path.basename(path).split('.')[0]
                
                current_file_samples = 0
                # Assuming structure is data_dict['train'][writer_id] = [samples]
                for writer_id, samples in data_dict['train'].items():
                    for sample in samples:
                        self.data.append(sample)
                        self.writer_ids.append(f"{prefix}_{writer_id}")
                        current_file_samples += 1
                print(f"      Loaded {current_file_samples} samples from {path}.")
            except Exception as e:
                print(f"      ❌ Failed to load {path}: {e}")

        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            self.writer_ids = self.writer_ids[:max_samples]
        
        # Create writer ID mapping
        unique_writers = sorted(set(self.writer_ids))
        self.writer_to_idx = {writer: idx for idx, writer in enumerate(unique_writers)}
        self.idx_to_writer = {idx: writer for idx, writer in enumerate(unique_writers)}
        self.num_writers = len(unique_writers)
        
        print(f"✅ Total: {len(self.data)} samples from {self.num_writers} unique writers (Combined)")
        
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
    print("\n" + "="*60)
    print("🔧 Pre-computing writer style features...")
    print("="*60)
    
    style_extractor.eval()
    writer_style_cache = {}  # {writer_idx (int): style_features}
    
    # Group samples by writer_idx
    writer_samples = {}
    for idx in range(len(dataset)):
        writer_id = dataset.writer_ids[idx]
        writer_idx = dataset.writer_to_idx[writer_id]
        
        if writer_idx not in writer_samples:
            writer_samples[writer_idx] = []
        writer_samples[writer_idx].append(idx)
    
    with torch.no_grad():
        for writer_idx, sample_indices in tqdm(writer_samples.items(), desc="Processing writers"):
            num_samples = min(samples_per_writer, len(sample_indices))
            selected_indices = np.random.choice(sample_indices, size=num_samples, replace=False)
            
            style_features_list = []
            for idx in selected_indices:
                sample = dataset.data[idx]
                img = sample['img']
                img = dataset.preprocess_image(img)
                if dataset.transforms:
                    img = dataset.transforms(img)
                img = img.unsqueeze(0).to(args.device)
                
                feat = style_extractor(img)
                style_features_list.append(feat.cpu())
            
            writer_style_cache[int(writer_idx)] = torch.stack(style_features_list).squeeze(1)
    
    print(f"✅ Cached style features for {len(writer_style_cache)} writers")
    return writer_style_cache


def collate_fn_with_cache(batch, style_cache, num_style_images=5):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])
    
    batch_style_features = []
    
    for writer_idx_tensor in writer_ids:
        writer_idx = int(writer_idx_tensor.item())
        
        if writer_idx not in style_cache:
            # Fallback or error handling
            raise KeyError(f"Writer index {writer_idx} not found in cache.")
        
        cached_features = style_cache[writer_idx]
        
        if cached_features.size(0) >= num_style_images:
            indices = np.random.choice(cached_features.size(0), size=num_style_images, replace=False)
        else:
            indices = np.random.choice(cached_features.size(0), size=num_style_images, replace=True)
        
        sampled_features = cached_features[indices]
        batch_style_features.append(sampled_features)
    
    batch_style_features = torch.stack(batch_style_features)
    return images, labels, writer_ids, batch_style_features

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])
    style_images = torch.stack([item[3] for item in batch])
    return images, labels, writer_ids, style_images

# ===== HELPER FUNCTIONS =====
def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def make_comparison_row(ref_tensors, gen_tensors, img_height=64, gap=10):
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
        if old is None: return new
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

# ===== TRAINING FUNCTION =====
def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, 
          num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, 
          tokenizer=None, text_encoder=None, lr_scheduler=None, use_cached_features=False):
    
    model.train()
    loss_meter = AvgMeter()
    print('\n🚀 Training started....')
    
    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'📅 Epoch: {epoch+1}/{args.epochs}')
        print(f'{"="*60}')
        
        # Cooling Break
        if epoch > 0 and epoch % args.cooling_interval == 0 and args.cooling_interval > 0:
            print(f"🧊 COOLING BREAK: Waiting {args.cooling_duration}s...")
            torch.cuda.empty_cache()
            time.sleep(args.cooling_duration)
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        epoch_start_time = time.time()
        
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_data = data[3]
            
            # Text Features
            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens) 
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long() 
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(transcr, padding="max_length", truncation=True, 
                                          return_tensors="pt", max_length=40).to(args.device)
            
            # Style Features
            if use_cached_features:
                style_features = style_data.to(args.device)
                B, N, feat_dim = style_features.shape
                style_features = style_features.reshape(B * N, feat_dim)
            else:
                if style_extractor is not None:
                    reshaped_images = style_data.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(reshaped_images)
                else:
                    style_features = None

            # VAE Encoding
            if args.latent == True:
                with torch.no_grad():
                    # Check if using DataParallel or not
                    if hasattr(vae, "module"):
                        images = vae.module.encode(images.to(torch.float32)).latent_dist.sample()
                    else:
                        images = vae.encode(images.to(torch.float32)).latent_dist.sample()
                        
                    images = images * 0.18215
            
            # Diffusion Steps
            noise = torch.randn(images.shape).to(images.device)
            timesteps = torch.randint(0, diffusion.noise_steps, (images.shape[0],), device=images.device).long()
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
            
            # Classifier-Free Guidance
            labels = s_id
            if np.random.random() < 0.1:
                labels = None
            
            predicted_noise = model(noisy_images, timesteps=timesteps, context=text_features, y=labels, style_extractor=style_features)
            loss = mse_loss(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.step_ema(ema_model, model)

            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(MSE=loss_meter.avg)
        
        epoch_time = time.time() - epoch_start_time
        print(f"📊 Avg Loss: {loss_meter.avg:.4f} | Time: {epoch_time:.2f}s")
        if lr_scheduler is not None:
            lr_scheduler.step()
            print(f"📉 LR: {lr_scheduler.get_last_lr()[0]:.8f}")

        # === MULTILINGUAL VISUALIZATION ===
        if epoch % 2 == 0:
            print(f"\n📸 Generating multilingual comparison...")
            torch.cuda.empty_cache()
            
            # MIXED LANGUAGE TEST: Bengali, English, Bengali
            hardcoded_sentence = 'সে books পড়ে' 
            gen_words = hardcoded_sentence.split()
            n_gen_words = len(gen_words)
            
            try:
                test_data = next(iter(test_loader))
            except StopIteration:
                test_data = next(iter(test_loader))
                
            batch_imgs = test_data[0]
            batch_writer_ids = test_data[2]
            batch_styles = test_data[3]
            
            unique_writers = torch.unique(batch_writer_ids)
            num_writers_to_log = min(5, len(unique_writers))
            row_images = []

            for idx in range(num_writers_to_log):
                writer_id = unique_writers[idx]
                indices = (batch_writer_ids == writer_id).nonzero(as_tuple=True)[0]
                selected_indices = indices[:4] 
                real_word_tensors = [batch_imgs[sidx] for sidx in selected_indices]
                
                # Style Inputs
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

                # Text Inputs
                if args.model_name == 'wordstylist':
                    batch_word_embeddings = []
                    for word in gen_words:
                        word_embedding = label_padding(word, num_tokens)
                        word_embedding = torch.from_numpy(np.array(word_embedding, dtype="int64")).long()
                        batch_word_embeddings.append(word_embedding)
                    text_features_gen = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    text_features_gen = tokenizer(gen_words, padding="max_length", truncation=True, 
                                                  return_tensors="pt", max_length=200).to(args.device)

                # Generation Loop
                ema_model.eval()
                noise_scheduler.set_timesteps(50)
                
                if args.latent:
                    x = torch.randn((n_gen_words, 4, 64 // 8, 256 // 8)).to(args.device)
                else:
                    x = torch.randn((n_gen_words, 3, 64, 256)).to(args.device)

                with torch.no_grad():
                    for time2 in noise_scheduler.timesteps:
                        t = (torch.ones(n_gen_words) * time2.item()).long().to(args.device)
                        noisy_residual = ema_model(x, t, text_features_gen, writer_id_input, style_extractor=style_features_gen)
                        x = noise_scheduler.step(noisy_residual, time2, x).prev_sample
                
                # ... inside the validation loop ...
                
                if args.latent:
                    latents = 1 / 0.18215 * x
                    
                    # === FIX START ===
                    # Check if VAE is wrapped in DataParallel (has .module) or not
                    if hasattr(vae, "module"):
                        image = vae.module.decode(latents).sample
                    else:
                        image = vae.decode(latents).sample
                    # === FIX END ===
                    
                else:
                    image = x
                
                generated_word_tensors = [image[k] for k in range(n_gen_words)]
                row_img_np = make_comparison_row(real_word_tensors, generated_word_tensors)
                row_images.append(row_img_np)
            
            if row_images:
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
                save_path = os.path.join(args.save_path, 'images', f"{epoch}_comparison.jpg")
                final_pil.save(save_path)
                print(f"✅ Saved multilingual comparison to {save_path}")
                if args.wandb_log:
                    experiment.log_image(final_pil, name=f"epoch_{epoch}_comparison", step=epoch)

        # Save Checkpoints
        print(f"💾 Saving models...")
        torch.save(model.state_dict(), os.path.join(args.save_path, "models", f"ckpt.pt"))
        torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", f"optim.pt"))


# ===== MAIN =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    # === MISSING ARGUMENT ADDED HERE ===
    parser.add_argument('--interpolation', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    
    # === CHANGED: Accept list of paths ===
    parser.add_argument('--pickle_path', type=str, default='./bengali_dataset.pickle',
                        help='Comma separated paths: ./ben.pkl,./eng.pkl')
    parser.add_argument('--max_samples', type=int, default=None)
    
    parser.add_argument('--cache_style_features', type=bool, default=True)
    parser.add_argument('--style_cache_path', type=str, default='./style_cache/writer_styles.pt')
    parser.add_argument('--num_style_samples', type=int, default=10)
    parser.add_argument('--num_style_images', type=int, default=5)
    
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_multi_model') 
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--style_path', type=str, default='./style_models/bengali_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='runwayml/stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--cooling_interval', type=int, default=3)
    parser.add_argument('--cooling_duration', type=int, default=10)
    parser.add_argument('--power_limit', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.wandb_log:
        global experiment
        experiment = Experiment(api_key="6xk1Nmcm6P2OmkiUlYSqe4IqV", project_name="diffusionpen-multi", workspace="rabib-jahin", auto_output_logging="simple")
    
    setup_logging(args)
    transform = transforms.Compose([transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Parse pickle paths
    pickle_paths_list = [p.strip() for p in args.pickle_path.split(',')]
    
    print('📦 Loading Multilingual Dataset...')
    full_dataset = PickleDataset(
        pickle_paths=pickle_paths_list,
        fixed_size=(64, 256),
        transforms=transform,
        args=args,
        max_samples=args.max_samples,
        load_style_images=not args.cache_style_features
    )
    
    style_classes = full_dataset.num_writers
    print(f'✅ Combined Style Classes: {style_classes}')
    
    # Tokenizer
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    device_ids = [0] if not args.dataparallel else [0, 1] # Simplified
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    if args.dataparallel:
        text_encoder = nn.DataParallel(text_encoder)
    text_encoder = text_encoder.to(args.device)

    # Feature Extractor
    print('📥 Loading style extractor...')
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    if os.path.exists(args.style_path):
        state_dict = torch.load(args.style_path, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        matched_keys = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(matched_keys)
        feature_extractor.load_state_dict(model_dict, strict=False)
        print("✅ Loaded style weights")
    
    if args.dataparallel:
        feature_extractor = DataParallel(feature_extractor)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.train()

    # Caching
    writer_style_cache = None
    if args.cache_style_features:
        if not os.path.exists(args.style_cache_path):
            writer_style_cache = precompute_writer_style_features(full_dataset, feature_extractor, args, args.num_style_samples)
            torch.save(writer_style_cache, args.style_cache_path)
        else:
            print(f"📂 Loading cache from {args.style_cache_path}")
            writer_style_cache = torch.load(args.style_cache_path)

    # DataLoaders
    test_size = args.batch_size
    train_data, test_data = random_split(full_dataset, [len(full_dataset) - test_size, test_size])
    
    collate = lambda b: collate_fn_with_cache(b, writer_style_cache, args.num_style_images) if writer_style_cache else collate_fn
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=args.num_workers)

    # UNet
    unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim, 
                     out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1, 1), 
                     channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, 
                     vocab_size=vocab_size, text_encoder=text_encoder, args=args)
    
    if args.dataparallel:
        unet = DataParallel(unet)
    unet = unet.to(args.device)

    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    
    global start_epoch
    if args.load_check:
        print('📥 Loading Checkpoint...')
        ckpt_num = 116 # Example
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        start_epoch = 19
        print(f'✅ Resuming from {start_epoch}')

    lr_scheduler = None
    
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
    if args.dataparallel:
        vae = DataParallel(vae)
    vae = vae.to(args.device).requires_grad_(False)
    
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, 
          style_classes, feature_extractor if not writer_style_cache else None, vocab_size, ddim, transform, args, 
          tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler, use_cached_features=writer_style_cache is not None)

if __name__ == "__main__":
    main()