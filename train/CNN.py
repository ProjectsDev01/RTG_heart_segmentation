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

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")  # zakładamy, że maski są tu przechowywane
IMG_SIZE = (256, 256)
BATCH_SIZE = 8  # zmniejszony rozmiar batcha
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
            landmark_path = os.path.join(landmark_dir, f.replace('.png', '.npy'))
            if not os.path.exists(landmark_path):
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
        points = np.load(landmark_path, allow_pickle=True)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=1)
        # Przetwarzanie
        image = cv2.resize(image, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        # Augmentacja (opcjonalnie)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        # Dodajemy wymiar kanału
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

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
    
    return accuracy.item(), dice.item(), iou.item()

# --- MODEL CNN DO SEGMENTACJI Z DROPOUT ---
class SegmentationCNN(nn.Module):
    def __init__(self):
        super(SegmentationCNN, self).__init__()
        # Encoder z dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        
        # Bottleneck z dropout
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1)  # Dodany dropout
        )
        
        # Decoder z dropout
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.Dropout(0.1)  # Dodany dropout
        )
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
    
    def forward(self, x):
        # Forward pass pozostaje bez zmian
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.bottleneck(x3)
        x = self.up1(x4)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.out_conv(x)
        return x
    
# --- FUNKCJA STRATY ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, inputs, targets, smooth=1e-6):
        inputs_sig = torch.sigmoid(inputs)
        intersection = (inputs_sig * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_sig.sum() + targets.sum() + smooth)
        return self.bce(inputs, targets) + dice_loss

# --- PRZYGOTOWANIE DANYCH I AUGMENTACJA ---
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.RandomGamma(gamma_limit=(80, 120)), 
    A.GaussianBlur(blur_limit=3, p=0.1)
])
full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR, transform=train_transform)

# Podział na zbiory: 80% train, 10% val, 10% test
indices = list(range(len(full_dataset)))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)
train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=BATCH_SIZE, pin_memory=True)
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, pin_memory=True)

# --- INICJALIZACJA MODELU, OPTIMIZERA, SCHEDULERA, STRATY ---
model = SegmentationCNN().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
criterion = DiceBCELoss()

# --- TRENING ---
history = {'train': {'loss': [], 'acc': [], 'dice': [], 'iou': []},
           'val': {'loss': [], 'acc': [], 'dice': [], 'iou': []}}
best_dice = 0
for epoch in range(EPOCHS):
    start_time = time.time()
    model.train()
    train_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        acc, dice, iou = calculate_metrics(outputs, masks)
        train_metrics['loss'] += loss.item()
        train_metrics['acc'] += acc
        train_metrics['dice'] += dice
        train_metrics['iou'] += iou
    model.eval()
    val_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, masks)
            acc, dice, iou = calculate_metrics(outputs, masks)
            val_metrics['loss'] += loss.item()
            val_metrics['acc'] += acc
            val_metrics['dice'] += dice
            val_metrics['iou'] += iou
    avg_val_dice = val_metrics['dice'] / len(val_loader)
    scheduler.step(avg_val_dice)
    for k in train_metrics:
        history['train'][k].append(train_metrics[k] / len(train_loader))
        history['val'][k].append(val_metrics[k] / len(val_loader))
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    print(f"Czas: {time.time()-start_time:.1f}s")
    print(f"Trening - Loss: {history['train']['loss'][-1]:.4f} | Dice: {history['train']['dice'][-1]:.4f}")
    print(f"Walidacja - Loss: {history['val']['loss'][-1]:.4f} | Dice: {history['val']['dice'][-1]:.4f}")
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "best_cnn_model.pth")

model.load_state_dict(torch.load("best_cnn_model.pth"))
model.eval()
test_metrics = {'loss': 0, 'acc': 0, 'dice': 0, 'iou': 0}
with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        acc, dice, iou = calculate_metrics(outputs, masks)
        test_metrics['loss'] += loss.item()
        test_metrics['acc'] += acc
        test_metrics['dice'] += dice
        test_metrics['iou'] += iou

print("\nWyniki testowe:")
print(f"Loss: {test_metrics['loss']/len(test_loader):.4f}")
print(f"Dice: {test_metrics['dice']/len(test_loader):.4f}")
print(f"IoU: {test_metrics['iou']/len(test_loader):.4f}")

plt.figure(figsize=(15,10))
metrics = ['loss', 'dice', 'iou', 'acc']
for i, metric in enumerate(metrics):
    plt.subplot(2,2,i+1)
    plt.plot(history['train'][metric], label='Trening')
    plt.plot(history['val'][metric], label='Walidacja')
    plt.title(metric.upper())
    plt.legend()
plt.tight_layout()
plt.savefig('cnn_metrics.png')
plt.show()
