import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import time
from medpy.metric.binary import hd, hd95
import torch.nn.functional as F

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")
IMG_SIZE = (256, 256)
BATCH_SIZE = 4

# Zwiększona liczba epok i zmniejszony learning rate
EPOCHS = 50
LR = 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GRAD_CLIP = 1.0  # maksymalna norma gradientu

# --- KLASA DATASETU ---
class HeartDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, files=None, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        
        if files is None:
            self.files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        else:
            self.files = files
        
        for f in self.files:
            landmark_file = f.replace('.png', '.npy')
            if not os.path.exists(os.path.join(landmark_dir, landmark_file)):
                raise FileNotFoundError(f"Brak pliku: {landmark_file}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        
        # Wczytywanie obrazu
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Wczytywanie maski
        landmark_path = os.path.join(self.landmark_dir, img_name.replace('.png', '.npy'))
        points = np.load(landmark_path)
        
        # Generowanie maski
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=1)
        
        # Zmiana rozmiaru (lub augmentacja)
        image = cv2.resize(image, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

# --- MODELE ---
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        
        self.theta = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        
        self.out_conv = nn.Conv2d(in_channels // 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        batch_size = x.size(0)
        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)
        
        theta = theta.view(batch_size, self.in_channels // 2, -1).permute(0, 2, 1)
        phi = phi.view(batch_size, self.in_channels // 2, -1)
        sim = torch.matmul(theta, phi) / (self.in_channels // 2) ** 0.5
        attn = torch.softmax(sim, dim=-1)
        
        g = g.view(batch_size, self.in_channels // 2, -1).permute(0, 2, 1)
        y = torch.matmul(attn, g)
        y = y.permute(0, 2, 1).view(batch_size, self.in_channels // 2, *x.shape[2:])
        y = self.out_conv(y)
        y = self.bn(y)
        return x + y
    
class MultiScaleHybridGNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_block(in_ch, out_ch, dropout=0.1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(dropout),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        
        # Encoder
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Middle with Non-Local Block
        self.non_local = NonLocalBlock(256)
        self.middle = conv_block(256, 512)
        
        # Decoder with Multi-Scale Aggregation
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(704, 256)  # 256 + 256 + 128 + 64
        
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(320, 128)   # 128 + 128 + 64
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)    # 64 + 64
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)          # (B, 64, H, W)
        e2 = self.enc2(self.pool(e1))  # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))  # (B, 256, H/4, W/4)
        
        # Middle
        m = self.pool(e3)           # (B, 256, H/8, W/8)
        m = self.non_local(m)       # Non-local block
        m = self.middle(m)          # (B, 512, H/8, W/8)
        
        # Decoder with Multi-Scale
        d3 = self.up3(m)            # (B, 256, H/4, W/4)
        
        # Pobierz rozmiar przestrzenny d3
        target_size = d3.shape[2:]
        
        # Skaluj cechy enkodera do rozmiaru d3
        e3_resized = e3  # (B, 256, H/4, W/4) - już pasuje
        e2_resized = F.interpolate(e2, size=target_size, mode='bilinear', align_corners=False)  # (B, 128, H/4, W/4)
        e1_resized = F.interpolate(e1, size=target_size, mode='bilinear', align_corners=False)  # (B, 64, H/4, W/4)
        
        # Konkatenacja wszystkich cech
        d3 = torch.cat([d3, e3_resized, e2_resized, e1_resized], dim=1)  # (B, 704, H/4, W/4)
        d3 = self.dec3(d3)          # (B, 256, H/4, W/4)
        
        # Krok dekodera dla d2
        d2 = self.up2(d3)           # (B, 128, H/2, W/2)
        target_size = d2.shape[2:]
        
        e2_resized = F.interpolate(e2, size=target_size, mode='bilinear', align_corners=False)  # (B, 128, H/2, W/2)
        e1_resized = F.interpolate(e1, size=target_size, mode='bilinear', align_corners=False)  # (B, 64, H/2, W/2)
        
        d2 = torch.cat([d2, e2_resized, e1_resized], dim=1)  # (B, 320, H/2, W/2)
        d2 = self.dec2(d2)          # (B, 128, H/2, W/2)
        
        # Krok dekodera dla d1
        d1 = self.up1(d2)           # (B, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)  # (B, 128, H, W)
        d1 = self.dec1(d1)          # (B, 64, H, W)
        
        return self.final(d1)       # (B, 1, H, W)
    
# --- METRYKI ---
def calculate_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    total_hd = 0.0
    total_hd95 = 0.0
    count = 0
    
    pred_bin_np = pred_bin.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)
    
    batch_size = pred_bin_np.shape[0]
    max_hd = np.sqrt(IMG_SIZE[0]**2 + IMG_SIZE[1]**2)
    
    for i in range(batch_size):
        pred_mask = pred_bin_np[i, 0]
        target_mask = target_np[i, 0]
        
        if np.sum(target_mask) == 0:
            continue
        
        count += 1
        
        if np.sum(pred_mask) == 0:
            total_hd += max_hd
            total_hd95 += max_hd
            continue
        
        try:
            current_hd = hd(pred_mask, target_mask)
            current_hd95 = hd95(pred_mask, target_mask)
        except:
            current_hd = max_hd
            current_hd95 = max_hd
        
        total_hd += min(current_hd, max_hd)
        total_hd95 += min(current_hd95, max_hd)
    
    avg_hd = total_hd / count if count > 0 else 0.0
    avg_hd95 = total_hd95 / count if count > 0 else 0.0
    
    return accuracy.item(), dice.item(), iou.item(), avg_hd, avg_hd95

# --- FUNKCJA STRATY ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets):
        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig * targets).sum()
        dice_loss = 1 - (2. * intersection) / (inputs_sig.sum() + targets.sum() + 1e-6)
        return self.bce(inputs, targets) + dice_loss

# --- PRZYGOTOWANIE DANYCH ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.RandomScale(scale_limit=0.1, p=0.2),
    A.RandomGamma(gamma_limit=(80, 120)),
    A.GaussianBlur(blur_limit=3, p=0.1),
    A.Resize(IMG_SIZE[0], IMG_SIZE[1])
])
val_transform = A.Compose([
    A.Resize(IMG_SIZE[0], IMG_SIZE[1])
])

all_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

train_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR, files=train_files, transform=train_transform)
val_dataset   = HeartDataset(IMG_DIR, LANDMARK_DIR, files=val_files, transform=val_transform)
test_dataset  = HeartDataset(IMG_DIR, LANDMARK_DIR, files=test_files, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# --- INICJALIZACJA MODELU ---
model = MultiScaleHybridGNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

# Zwiększone patience w schedulerze (z 3 do 5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)

criterion = DiceBCELoss()

# --- TRENING ---
history = {
    'train': {'loss': [], 'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []},
    'val': {'loss': [], 'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []}
}

best_dice = 0
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    train_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"\nEpoch {epoch+1}/{EPOCHS} - LR: {current_lr:.6f}")
    
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        
        acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
        train_metrics['loss'] += loss.item()
        train_metrics['acc'] += acc
        train_metrics['dice'] += dice
        train_metrics['iou'] += iou
        train_metrics['hd'] += hd_val
        train_metrics['hd95'] += hd95_val
    
    model.eval()
    val_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
            
            val_metrics['loss'] += loss.item()
            val_metrics['acc'] += acc
            val_metrics['dice'] += dice
            val_metrics['iou'] += iou
            val_metrics['hd'] += hd_val
            val_metrics['hd95'] += hd95_val
    
    avg_val_dice = val_metrics['dice'] / len(val_loader)
    scheduler.step(avg_val_dice)
    
    for k in ['loss', 'acc', 'dice', 'iou', 'hd', 'hd95']:
        history['train'][k].append(train_metrics[k] / len(train_loader))
        history['val'][k].append(val_metrics[k] / len(val_loader))
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Czas: {time.time()-start_time:.1f}s")
    print(f"Trening - Loss: {history['train']['loss'][-1]:.4f} | Dice: {history['train']['dice'][-1]:.4f} | HD: {history['train']['hd'][-1]:.2f} | HD95: {history['train']['hd95'][-1]:.2f}")
    print(f"Walidacja - Loss: {history['val']['loss'][-1]:.4f} | Dice: {history['val']['dice'][-1]:.4f} | HD: {history['val']['hd'][-1]:.2f} | HD95: {history['val']['hd95'][-1]:.2f}")
    
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "mshybridgnet_model.pth")

# --- TEST ---
model.load_state_dict(torch.load("mshybridgnet_model.pth"))
model.eval()
test_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        
        loss = criterion(outputs, masks)
        acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
        
        test_metrics['loss'] += loss.item()
        test_metrics['acc'] += acc
        test_metrics['dice'] += dice
        test_metrics['iou'] += iou
        test_metrics['hd'] += hd_val
        test_metrics['hd95'] += hd95_val

print("\nWyniki testowe:")
print(f"Loss: {test_metrics['loss']/len(test_loader):.4f}")
print(f"Dice: {test_metrics['dice']/len(test_loader):.4f}")
print(f"IoU: {test_metrics['iou']/len(test_loader):.4f}")
print(f"HD: {test_metrics['hd']/len(test_loader):.2f}")
print(f"HD95: {test_metrics['hd95']/len(test_loader):.2f}")

# --- WIZUALIZACJA ---
plt.figure(figsize=(20, 15))
metrics = ['loss', 'dice', 'iou', 'hd', 'hd95']
for i, metric in enumerate(metrics):
    plt.subplot(3, 2, i+1)
    plt.plot(history['train'][metric], label='Trening')
    plt.plot(history['val'][metric], label='Walidacja')
    plt.title(metric.upper())
    plt.legend()
plt.tight_layout()
plt.savefig('mshybridgnet_metrics.png')
