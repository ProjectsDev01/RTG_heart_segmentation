import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
import time
from medpy.metric.binary import hd, hd95

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")
IMG_SIZE = (256, 256)
BATCH_SIZE = 4  # Zmniejszony batch size dla lepszego zarządzania pamięcią
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- KLASA DATASETU ---
class HeartDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        
        # Weryfikacja spójności danych
        for f in self.files:
            if not os.path.exists(os.path.join(landmark_dir, f.replace('.png', '.npy'))):
                raise FileNotFoundError(f"Brak pliku: {f.replace('.png', '.npy')}")

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
        
        # Przetwarzanie
        image = cv2.resize(image, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Augmentacja
        if self.transform:
            augmented = self.transform(
                image=image,
                mask=mask
            )
            image = augmented['image']
            mask = augmented['mask']
        
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

# --- METRYKI ---
def calculate_metrics(pred, target):
    pred = torch.sigmoid(pred)
    pred_bin = (pred > 0.5).float()
    
    # Tradycyjne metryki
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1-target)).sum()
    fn = ((1-pred_bin) * target).sum()
    tn = ((1-pred_bin) * (1-target)).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    
    # Metryki geometryczne
    total_hd = 0.0
    total_hd95 = 0.0
    count = 0
    
    pred_bin_np = pred_bin.cpu().numpy().astype(np.uint8)
    target_np = target.cpu().numpy().astype(np.uint8)
    
    batch_size = pred_bin_np.shape[0]
    max_hd = np.sqrt(IMG_SIZE[0]**2 + IMG_SIZE[1]**2)  # Maksymalna możliwa odległość
    
    for i in range(batch_size):
        pred_mask = pred_bin_np[i, 0]
        target_mask = target_np[i, 0]
        
        if np.sum(target_mask) == 0:
            continue  # Pomijamy przypadki bez targetu
        
        count += 1
        
        if np.sum(pred_mask) == 0:
            # Predykcja pusta przy istniejącym targecie
            total_hd += max_hd
            total_hd95 += max_hd
            continue
        
        try:
            current_hd = hd(pred_mask, target_mask)
            current_hd95 = hd95(pred_mask, target_mask)
        except:
            current_hd = max_hd
            current_hd95 = max_hd
        
        # Ograniczenie do maksymalnej wartości
        current_hd = min(current_hd, max_hd)
        current_hd95 = min(current_hd95, max_hd)
        
        total_hd += current_hd
        total_hd95 += current_hd95
    
    avg_hd = total_hd / count if count > 0 else 0.0
    avg_hd95 = total_hd95 / count if count > 0 else 0.0
    
    return accuracy.item(), dice.item(), iou.item(), avg_hd, avg_hd95

# --- MODEL U-NET z Dropout ---
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.2),  # Dodany dropout
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2)   # Dodany dropout
            )
        
        # Encoder
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Middle
        self.middle = conv_block(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)
        
        self.final = nn.Conv2d(64, 1, kernel_size=1)
        
        # Inicjalizacja wag
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # Middle
        m = self.middle(self.pool(e3))
        
        # Decoder
        d3 = self.up3(m)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

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

full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR, train_transform)

# Podział na zbiory
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, pin_memory=True)
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, pin_memory=True)

# --- INICJALIZACJA ---
model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # Zwiększona regularyzacja
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
criterion = DiceBCELoss()

# --- TRENING ---
history = {
    'train': {'loss': [], 'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []},
    'val': {'loss': [], 'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []}
}

best_dice = 0
for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Faza treningowa
    model.train()
    train_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
        train_metrics['loss'] += loss.item()
        train_metrics['acc'] += acc
        train_metrics['dice'] += dice
        train_metrics['iou'] += iou
        train_metrics['hd'] += hd_val
        train_metrics['hd95'] += hd95_val
    
    # Faza walidacyjna
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
    
    # Aktualizacja scheduler
    avg_val_dice = val_metrics['dice']/len(val_loader)
    scheduler.step(avg_val_dice)
    
    # Zapis wyników
    for k in ['loss', 'acc', 'dice', 'iou', 'hd', 'hd95']:
        history['train'][k].append(train_metrics[k]/len(train_loader))
        history['val'][k].append(val_metrics[k]/len(val_loader))
    
    # Wydruk informacji
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Czas: {time.time()-start_time:.1f}s")
    print(f"Trening - Loss: {history['train']['loss'][-1]:.4f} | Dice: {history['train']['dice'][-1]:.4f} | HD: {history['train']['hd'][-1]:.2f} | HD95: {history['train']['hd95'][-1]:.2f}")
    print(f"Walidacja - Loss: {history['val']['loss'][-1]:.4f} | Dice: {history['val']['dice'][-1]:.4f} | HD: {history['val']['hd'][-1]:.2f} | HD95: {history['val']['hd95'][-1]:.2f}")
    
    # Zapis modelu
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "unet_model50eb4lr4drop1.pth")

# --- TEST ---
model.load_state_dict(torch.load("unet_model50eb4lr4drop1.pth"))
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
plt.savefig('unet_metrics50eb4lr4drop1.png')