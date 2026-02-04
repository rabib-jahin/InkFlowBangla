import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from PIL import Image, ImageOps
import os
import argparse
import torch.optim as optim
from tqdm import tqdm
from src.models.feature_extractor import ImageEncoder
import time
import pickle
import random


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


class PickleStyleDataset(Dataset):
    """
    Custom Dataset for loading Bengali text images from pickle file
    Specifically designed for style encoder training with triplet loss
    """
    
    def __init__(self, pickle_path, fixed_size=(64, 256), transforms=None, max_samples=None):
        self.fixed_size = fixed_size
        self.transforms = transforms
        
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
        
        # Group samples by writer for efficient sampling
        self.writer_samples = {}
        for idx, writer_id in enumerate(self.writer_ids):
            if writer_id not in self.writer_samples:
                self.writer_samples[writer_id] = []
            self.writer_samples[writer_id].append(idx)
        
        if self.transforms is None:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        return len(self.data)
    
    def preprocess_image(self, img):
        """Preprocess PIL image to fixed size with padding"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_width, img_height = img.size
        target_height, target_width = self.fixed_size
        
        # Resize maintaining aspect ratio based on height
        if img_height != target_height:
            scale = target_height / img_height
            new_width = int(img_width * scale)
            img = img.resize((new_width, target_height), Image.LANCZOS)
            img_width = new_width
        
        # Pad or crop to target width
        if img_width < target_width:
            img = ImageOps.pad(img, size=(target_width, target_height), color="white")
        elif img_width > target_width:
            left = (img_width - target_width) // 2
            img = img.crop((left, 0, left + target_width, target_height))
        
        return img
    
    def __getitem__(self, idx):
        """
        Returns:
            anchor: Main image
            transcr: Text label
            writer_idx: Writer ID as integer
            positive: Another image from the same writer
            negative: Image from a different writer
        """
        sample = self.data[idx]
        writer_id = self.writer_ids[idx]
        
        # Get anchor image and label
        anchor_img = sample['img']
        transcr = sample['label']
        
        # Preprocess anchor image
        anchor_img = self.preprocess_image(anchor_img)
        
        # Get writer ID index
        writer_idx = self.writer_to_idx[writer_id]
        
        # Get positive sample (same writer)
        positive_samples = self.writer_samples[writer_id]
        if len(positive_samples) > 1:
            positive_idx = random.choice([i for i in positive_samples if i != idx])
        else:
            positive_idx = idx  # Fallback if only one sample
        
        positive_img = self.data[positive_idx]['img']
        positive_img = self.preprocess_image(positive_img)
        
        # Get negative sample (different writer)
        other_writers = [w for w in self.writer_samples.keys() if w != writer_id]
        if len(other_writers) > 0:
            negative_writer = random.choice(other_writers)
            negative_idx = random.choice(self.writer_samples[negative_writer])
        else:
            negative_idx = idx  # Fallback
        
        negative_img = self.data[negative_idx]['img']
        negative_img = self.preprocess_image(negative_img)
        
        # Apply transforms
        if self.transforms:
            anchor_img = self.transforms(anchor_img)
            positive_img = self.transforms(positive_img)
            negative_img = self.transforms(negative_img)
        
        return anchor_img, transcr, torch.tensor(writer_idx, dtype=torch.long), positive_img, negative_img


def collate_fn(batch):
    """Custom collate function"""
    anchors = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]
    writer_ids = torch.stack([item[2] for item in batch])
    positives = torch.stack([item[3] for item in batch])
    negatives = torch.stack([item[4] for item in batch])
    
    return anchors, labels, writer_ids, positives, negatives


#================ Performance and Loss Function ========================
def performance(pred, label):
    loss = nn.CrossEntropyLoss()
    loss = loss(pred, label)
    return loss 


#===================== Training ==========================================
def train_epoch_triplet(train_loader, model, criterion, optimizer, device, args):
    model.train()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(train_loader)
    
    for i, data in enumerate(pbar):
        anchor = data[0].to(device)
        wid = data[2].to(device)
        positive = data[3].to(device)
        negative = data[4].to(device)

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        count = anchor.size(0)
        loss_meter.update(loss.item(), count)
        pbar.set_postfix(triplet_loss=loss_meter.avg)
        total += anchor.size(0)
    
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    return running_loss/total


def val_epoch_triplet(val_loader, model, criterion, device, args):
    model.eval()
    running_loss = 0
    total = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            anchor = data[0].to(device)
            wid = data[2].to(device)
            positive = data[3].to(device)
            negative = data[4].to(device)
        
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            
            loss = criterion(anchor_out, positive_out, negative_out)
            
            running_loss += loss.item()
            count = anchor.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(triplet_loss=loss_meter.avg)
            total += wid.size(0)
    
    print(f"Validation Loss: {running_loss/len(val_loader):.4f}")
    return running_loss/total


############################ MIXED TRAINING ############################################              
def train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args):
    model.train()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    loss_meter_triplet = AvgMeter()
    loss_meter_class = AvgMeter()
    pbar = tqdm(train_loader)
    
    for i, data in enumerate(pbar):
        anchor = data[0].to(device)
        wid = data[2].to(device)
        positive = data[3].to(device)
        negative = data[4].to(device)
        
        # Get logits and features from the model
        anchor_logits, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)
        
        _, preds = torch.max(anchor_logits.data, 1)
        n_corrects += (preds == wid.data).sum().item()
    
        classification_loss = performance(anchor_logits, wid)
        triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
        
        loss = classification_loss + triplet_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        count = anchor.size(0)
        loss_meter.update(loss.item(), count)
        loss_meter_triplet.update(triplet_loss.item(), count)
        loss_meter_class.update(classification_loss.item(), count)
        pbar.set_postfix(
            mixed_loss=loss_meter.avg, 
            classification_loss=loss_meter_class.avg, 
            triplet_loss=loss_meter_triplet.avg
        )
        total += anchor.size(0)
    
    accuracy = n_corrects/total
    print(f"Training Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training Accuracy: {accuracy*100:.4f}%")
    return running_loss/total


def val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, device, args):
    model.eval()
    running_loss = 0
    total = 0
    n_corrects = 0
    loss_meter = AvgMeter()
    pbar = tqdm(val_loader)
    
    with torch.no_grad():
        for i, data in enumerate(pbar):
            anchor = data[0].to(device)
            wid = data[2].to(device)
            positive = data[3].to(device)
            negative = data[4].to(device)
            
            anchor_logits, anchor_features = model(anchor)
            _, positive_features = model(positive)
            _, negative_features = model(negative)
            
            _, preds = torch.max(anchor_logits.data, 1)
            n_corrects += (preds == wid.data).sum().item()
        
            classification_loss = performance(anchor_logits, wid)
            triplet_loss = criterion_triplet(anchor_features, positive_features, negative_features)
            
            loss = classification_loss + triplet_loss
            
            running_loss += loss.item()
            count = anchor.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(mixed_loss=loss_meter.avg)
            total += wid.size(0)
    
    accuracy = n_corrects/total
    print(f"Validation Loss: {running_loss/len(val_loader):.4f}")
    print(f"Validation Accuracy: {accuracy*100:.4f}%")
    return running_loss/total


#TRAINING CALLS
def train_mixed(model, train_loader, val_loader, criterion_triplet, criterion_classification, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        print(f"Epoch: {epoch_i+1}/{args.epochs}")
        train_loss = train_epoch_mixed(train_loader, model, criterion_triplet, criterion_classification, optimizer, device, args)
        
        val_loss = val_epoch_mixed(val_loader, model, criterion_triplet, criterion_classification, device, args)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{args.save_path}/mixed_bengali_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)


def train_triplet(model, train_loader, val_loader, criterion, optimizer, scheduler, device, args):
    best_loss = float('inf')
    for epoch_i in range(args.epochs):
        print(f"Epoch: {epoch_i+1}/{args.epochs}")
        train_loss = train_epoch_triplet(train_loader, model, criterion, optimizer, device, args)
        
        val_loss = val_epoch_triplet(val_loader, model, criterion, device, args)
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{args.save_path}/triplet_bengali_{args.model}.pth')
            print("Saved Best Model!")
        
        scheduler.step(val_loss)


class Mixed_Encoder(nn.Module):
    """Encode images with both classification and embedding outputs"""
    def __init__(self, model_name='mobilenetv2_100', num_classes=339, pretrained=True, trainable=True):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="")
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        if hasattr(self.model, 'num_features'):
            num_features = self.model.num_features
        else:
            num_features = 2048

        self.classifier = nn.Linear(num_features, num_classes)

        for p in self.model.parameters():
            p.requires_grad = trainable
            
    def forward(self, x):
        features = self.model(x)
        pooled_features = self.global_pool(features).flatten(1)
        logits = self.classifier(pooled_features)
        return logits, pooled_features


def main():
    parser = argparse.ArgumentParser(description='Train Style Encoder on Pickle Dataset')
    parser.add_argument('--model', type=str, default='mobilenetv2_100', 
                        help='type of cnn to use (resnet18, mobilenetv2_100, etc.)')
    parser.add_argument('--pickle_path', type=str, required=True,
                        help='path to pickle dataset file')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='number of training epochs')
    parser.add_argument('--pretrained', type=bool, default=False, 
                        help='use pretrained feature extractor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training / testing')
    parser.add_argument('--save_path', type=str, default='./style_models', 
                        help='path to save models')
    parser.add_argument('--mode', type=str, default='mixed', 
                        help='mixed (DiffusionPen), triplet, or classification')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of data loading workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='maximum number of samples to load (None = all)')
    args = parser.parse_args()

    # Ensure device is properly set
    if not torch.cuda.is_available() and 'cuda' in args.device:
        print(f"WARNING: CUDA not available, switching from {args.device} to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    print('Loading Bengali pickle dataset...')
    full_dataset = PickleStyleDataset(
        pickle_path=args.pickle_path,
        fixed_size=(64, 256),
        transforms=train_transform,
        max_samples=args.max_samples
    )
    
    style_classes = full_dataset.num_writers
    print(f'Number of unique writers (style classes): {style_classes}')
    
    # Split into train and validation
    validation_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - validation_size
    
    train_data, val_data = random_split(
        full_dataset, 
        [train_size, validation_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f'Train samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    if args.mode == 'mixed':
        print(f'Using Mixed_Encoder with {args.model}')
        model = Mixed_Encoder(
            model_name=args.model, 
            num_classes=style_classes, 
            pretrained=True, 
            trainable=True
        )
    else:
        print(f'Using ImageEncoder with {args.model}')
        model = ImageEncoder(
            model_name=args.model, 
            num_classes=style_classes if args.mode == 'classification' else 0,
            pretrained=True, 
            trainable=True
        )
    
    print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    
    model = model.to(device)
    
    # Optimizer and scheduler
    optimizer_ft = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_ft, mode="min", patience=3, factor=0.1
    )
    
    # Loss functions
    criterion_triplet = nn.TripletMarginLoss(margin=1.0, p=2)
    
    # Training
    if args.mode == 'mixed':
        print('Using both classification and metric learning training')
        train_mixed(model, train_loader, val_loader, criterion_triplet, None, optimizer_ft, lr_scheduler, device, args)
    elif args.mode == 'triplet':
        print('Using triplet loss training')
        train_triplet(model, train_loader, val_loader, criterion_triplet, optimizer_ft, lr_scheduler, device, args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    
    print('Finished training!')


if __name__ == '__main__':
    main()