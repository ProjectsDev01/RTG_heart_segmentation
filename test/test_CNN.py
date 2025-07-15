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
from scipy.spatial.distance import directed_hausdorff, cdist  # for Hausdorff metrics

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DEFINICJA DATASETU ---
class HeartDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        for f in self.files:
            if not os.path.exists(os.path.join(landmark_dir, f.replace('.png', '.npy'))):
                raise FileNotFoundError(f"Brak pliku: {f.replace('.png', '.npy')}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        landmark_path = os.path.join(self.landmark_dir, img_name.replace('.png', '.npy'))
        points = np.load(landmark_path, allow_pickle=True)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=1)
        image = cv2.resize(image, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return image, mask

# --- METRYKI ---
def calculate_metrics(pred, target):
    pred_sig = torch.sigmoid(pred)
    pred_bin = (pred_sig > 0.5).float()
    tp = (pred_bin * target).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return acc.item(), dice.item(), iou.item(), pred_bin

# --- METRYKI HD i HD95 ---
def hausdorff_distance(pred_np, target_np):
    p_pts = np.argwhere(pred_np)
    t_pts = np.argwhere(target_np)
    if p_pts.size == 0 or t_pts.size == 0:
        return np.nan
    d1 = directed_hausdorff(p_pts, t_pts)[0]
    d2 = directed_hausdorff(t_pts, p_pts)[0]
    return max(d1, d2)

def hd95(pred_np, target_np):
    p_pts = np.argwhere(pred_np)
    t_pts = np.argwhere(target_np)
    if p_pts.size == 0 or t_pts.size == 0:
        return np.nan
    dists = cdist(p_pts, t_pts)
    d_p2t = dists.min(axis=1)
    d_t2p = dists.min(axis=0)
    return max(np.percentile(d_p2t, 95), np.percentile(d_t2p, 95))

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

criterion = DiceBCELoss()

# --- ARCHITEKTURA MODELU CNN DO SEGMENTACJI ---
class SegmentationCNN(nn.Module):
    def __init__(self):
        super(SegmentationCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.dec3(x)
        x = self.out_conv(x)
        x = nn.functional.interpolate(x, size=IMG_SIZE, mode='bilinear', align_corners=False)
        return x

# --- PRZYGOTOWANIE ZBIORU TESTOWEGO ---
full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR)
indices = list(range(len(full_dataset)))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- ZAŁADOWANIE MODELU ---
model = SegmentationCNN().to(DEVICE)
model.load_state_dict(torch.load("best_cnn_model.pth", map_location=DEVICE))
model.eval()

# --- TESTOWANIE I METRYKI ---
test_loss = test_acc = test_dice = test_iou = 0.0
sum_hd = sum_hd95 = 0.0
count = 0
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testowanie"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        acc, dice, iou, preds = calculate_metrics(outputs, masks)
        test_loss += loss.item()
        test_acc += acc
        test_dice += dice
        test_iou += iou
        preds_np = preds.cpu().numpy()
        masks_np = masks.cpu().numpy()
        for i in range(preds_np.shape[0]):
            hd = hausdorff_distance(preds_np[i,0], masks_np[i,0])
            hd95v = hd95(preds_np[i,0], masks_np[i,0])
            if not np.isnan(hd): sum_hd += hd
            if not np.isnan(hd95v): sum_hd95 += hd95v
            count += 1

# --- WYŚWIETLANIE WYNIKÓW ---
batches = len(test_loader)
print("\nWyniki na zbiorze testowym:")
print(f"Loss: {test_loss/batches:.4f}")
print(f"Accuracy: {test_acc/batches:.4f}")
print(f"Dice: {test_dice/batches:.4f}")
print(f"IoU: {test_iou/batches:.4f}")
if count>0:
    print(f"Hausdorff Distance (avg): {sum_hd/count:.4f}")
    print(f"Hausdorff95 (avg): {sum_hd95/count:.4f}")

# --- WIZUALIZACJA JEDNEJ PRÓBKI ---
images, masks = next(iter(test_loader))
images, masks = images.to(DEVICE), masks.to(DEVICE)
with torch.no_grad():
    outputs = model(images)
    preds = (torch.sigmoid(outputs) > 0.5).float()

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(images[0,0].cpu().numpy(), cmap='gray'); axs[0].set_title('Oryginalny obraz'); axs[0].axis('off')
axs[1].imshow(masks[0,0].cpu().numpy(), cmap='gray'); axs[1].set_title('Maska referencyjna'); axs[1].axis('off')
axs[2].imshow(preds[0,0].cpu().numpy(), cmap='gray'); axs[2].set_title('Maska przewidywana'); axs[2].axis('off')
plt.tight_layout(); plt.show()
