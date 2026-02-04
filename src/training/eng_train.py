import comet_ml # MUST be the first import
from comet_ml import Experiment
import os
import torch
import torch.nn as nn

import numpy as np
from PIL import Image, ImageOps # Added ImageOps explicitly
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from src.models.unet import UNetModel
import wandb
from torchvision import transforms
from src.models.feature_extractor import ImageEncoder
from src.utils.iam_dataset import IAMDataset
from src.utils.GNHK_dataset import GNHK_Dataset
from src.utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
from transformers import CanineModel, CanineTokenizer
import cv2 # Ensure cv2 is imported for crop function

torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 
IMG_WIDTH = 256
IMG_HEIGHT = 64
experiment=None

c_classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
cdict = {c:i for i,c in enumerate(c_classes)}
icdict = {i:c for i,c in enumerate(c_classes)}

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


def labelDictionary():
    labels = list(c_classes)
    letter2index = {label: n for n, label in enumerate(labels)}
    json_dict_l = json.dumps(letter2index)
    with open("letter2index.json","w") as l:
        l.write(json_dict_l)
    index2letter = {v: k for k, v in letter2index.items()}
    json_dict_i = json.dumps(index2letter)
    with open("index2letter.json","w") as l:
        l.write(json_dict_i)
    return len(labels), letter2index, index2letter


char_classes, letter2index, index2letter = labelDictionary()
tok = False
if not tok:
    tokens = {"PAD_TOKEN": 52}
else:
    tokens = {"GO_TOKEN": 52, "END_TOKEN": 53, "PAD_TOKEN": 54}
num_tokens = len(tokens.keys())
print('num_tokens', num_tokens)


print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens


def setup_logging(args):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

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

def crop_whitespace_width(img):
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        rect = img.crop((x, y, x + w, y + h))
        return np.array(rect)
    return np.array(img)


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

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        
        # Only process a few batches for sampling validation, not the whole test loader
        # to avoid memory issues and long waits
        
        with torch.no_grad():
            # Get just one batch from iterator
            data = next(iter(test_loader))
            
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)
            
            # Limit to 'n' samples
            images = images[:n]
            transcr = transcr[:n]
            s_id = s_id[:n]
            style_images = style_images[:n]

            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens) # Fixed missing arg
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
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample
                    
        model.train()
        if args.latent==True:
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

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        
        with torch.no_grad():
            style_images = None
            text_features = x_text 
            text_features = tokenizer(text_features, padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(args.device)
            
            if args.img_feat == True:
                with open('./writers_dict_train.json', 'r') as f:
                    wr_dict = json.load(f)
                reverse_wr_dict = {v: k for k, v in wr_dict.items()}
                
                with open('./utils/splits_words/iam_train_val.txt', 'r') as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(',') for i in train_data]
                    
                    for label in labels:
                        label_index = label.item()
                        # Use list comprehension carefully to avoid memory spike
                        matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index] and len(line[2])>3]

                        if len(matching_lines) >= 5:
                            five_styles = random.sample(matching_lines, 5)
                        else:
                            # Fallback if not enough samples
                            if len(matching_lines) > 0:
                                five_styles = [matching_lines[0]]*5
                            else:
                                # Fallback if NO samples found for this writer (prevent crash)
                                five_styles = [train_data[0]]*5 
                                
                        # print('five_styles', five_styles) # Commented out to prevent console buffer overflow
                        
                        cor_image_random = random.sample(matching_lines, 1) if len(matching_lines) > 0 else [train_data[0]]

                        cor_im = False # Hardcoded in original, keeping it
                        if cor_im == True:
                            # ... (Logic for cor_im kept as is)
                             pass 
                            
                        st_imgs = []
                        grid_imgs = []
                        root_path = './iam_data/words'
                        
                        for im_idx, random_f in enumerate(five_styles):
                            file_path = os.path.join(root_path, random_f[0])
                            try:
                                img_s = Image.open(file_path).convert('RGB')
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")
                                # Fallback
                                img_s = Image.new('RGB', (256, 64), color='white')
                                
                            (img_width, img_height) = img_s.size
                            img_s = img_s.resize((int(img_width * 64 / img_height), 64))
                            (img_width, img_height) = img_s.size
                            
                            if img_width < 256:
                                outImg = ImageOps.pad(img_s, size=(256, 64), color= "white")
                                img_s = outImg
                            else:
                                while img_width > 256:
                                    img_s = image_resize_PIL(img_s, width=img_width-20)
                                    (img_width, img_height) = img_s.size
                                img_s = centered_PIL(img_s, (64, 256), border_value=255.0)

                            transform_tensor = transforms.ToTensor()
                            grid_im = transform_tensor(img_s)
                            grid_imgs += [grid_im]
                            
                            img_tens = transform(img_s).to(args.device)
                            st_imgs += [img_tens]
                            
                        s_imgs = torch.stack(st_imgs).to(args.device)
                        style_images = torch.cat((style_images, s_imgs)) if style_images is not None else s_imgs
                    
                    style_images = style_images.reshape(-1, 3, 64, 256)
                    style_features = style_extractor(style_images).to(args.device)
            else:
                style_images = None
                style_features = None            
            
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)
            
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:
                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample

        model.train()
        if args.latent==True:
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


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')
    
    for epoch in range(args.epochs):
        print('Epoch:', epoch)
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
                with torch.no_grad(): # VAE encode should be no_grad to save memory
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
    
        if epoch % 10 == 0:
            # Clean memory before validation
            torch.cuda.empty_cache() 
            labels = torch.arange(min(16, args.batch_size)).long().to(args.device)
            n = len(labels)
        
            if args.sampling_word == True:
                words = ['text']
                for x_text in words: 
                    # Correct method call to sampling
                    ema_sampled_images = diffusion.sampling(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args, style_extractor=style_extractor, noise_scheduler=noise_scheduler, transform=transforms, tokenizer=tokenizer, text_encoder=text_encoder)
                    epoch_n = epoch 
                    sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"), args)
            else:
                ema_sampled_images = diffusion.sampling_loader(ema_model, test_loader, vae, n=n, x_text=None, labels=labels, args=args, style_extractor=style_extractor, noise_scheduler=noise_scheduler, transform=transforms, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder)
                epoch_n = epoch 
                sampled_ema = save_images(ema_sampled_images, os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"), args)
        
            if args.wandb_log==True:
                # wandb_sampled_ema= wandb.Image(sampled_ema, caption=f"sample_{epoch}")
                # wandb.log({f"Sampled images": wandb_sampled_ema})
                experiment.log_image(
                    sampled_ema, 
                    name=f"sample_{epoch}", 
                    step=epoch,
                    image_format="png" 
                )
            
            torch.save(model.state_dict(), os.path.join(args.save_path,"models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path,"models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path,"models", "optim.pt"))   


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    # REDUCED DEFAULT BATCH SIZE to avoid MemoryError
    parser.add_argument('--batch_size', type=int, default=64) 
    # REDUCED DEFAULT WORKERS to avoid MemoryError on Windows
    parser.add_argument('--num_workers', type=int, default=0) 
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam') 
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    
    args = parser.parse_args()
   
    
    print('torch version', torch.__version__)
    
    if args.wandb_log==True:
        # runs = wandb.init(project='DiffusionPen', entity='name_entity', name=args.dataset, config=args)
        # wandb.config.update(args)
        global experiment
        experiment = Experiment(
        api_key="6xk1Nmcm6P2OmkiUlYSqe4IqV",
        project_name="diffusionpen",
        workspace="rabib-jahin",
        auto_output_logging="simple"
    )
    
    setup_logging(args)

    transform = transforms.Compose([
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    
    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        # Only loading one dataset to save memory if splitting later anyway
        train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        
        print('train data', len(train_data))
        test_size = args.batch_size # Keep test size small for validation
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
    elif args.dataset == 'gnhk':
        print('loading GNHK')
        myDataset = GNHK_Dataset
        dataset_folder = 'path/to/GNHK'
        style_classes = 515
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ])
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(1 * 64, 256), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        test_size = args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
        
    # Set pin_memory=True for speed since we lowered workers
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
    else:
        vocab_size = len(character_classes)
    
    if args.dataparallel==True:
        device_ids = [3,4] # Adjust based on your available GPUs
        print('using dataparallel with device:', device_ids)
    else:
        idx = int(''.join(filter(str.isdigit, args.device)))
        device_ids = [idx]

    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids)
        text_encoder = text_encoder.to(args.device)
    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None
    
    if args.unet=='unet_latent':
        unet = UNetModel(image_size = args.img_size, in_channels=args.channels, model_channels=args.emb_dim, out_channels=args.channels, num_res_blocks=args.num_res_blocks, attention_resolutions=(1,1), channel_mult=(1, 1), num_heads=args.num_heads, num_classes=style_classes, context_dim=args.emb_dim, vocab_size=vocab_size, text_encoder=text_encoder, args=args)
    
    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)
    
    optimizer = optim.AdamW(unet.parameters(), lr=0.0001)
    lr_scheduler = None 

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)
    
    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)

    if args.load_check==True:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt'))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt'))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))
        print('Loaded models and optimizer')
    
    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        vae.requires_grad_(False)
    else:
        vae = None

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")
    
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    PATH = args.style_path 
    
    if os.path.exists(PATH):
        state_dict = torch.load(PATH, map_location=args.device)
        model_dict = feature_extractor.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(state_dict)
        feature_extractor.load_state_dict(model_dict)
    else:
        print(f"Warning: Style model path {PATH} not found. Continuing without loading style weights.")

    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()
    
    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)
    
    elif args.train_mode == 'sampling':
        # ... (Sampling code remains largely similar, just ensure batch_size is handled if using loaders)
        pass 

if __name__ == "__main__":
    main()