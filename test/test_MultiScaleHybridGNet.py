import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.files = []
        for f in os.listdir(image_dir):
            if f.endswith('.png'):
                landmark_name = f.replace('.png', '.npy')
                if os.path.exists(os.path.join(landmark_dir, landmark_name)):
                    self.files.append(f)
                else:
                    print(f"Warning: Brak pliku {landmark_name}, pomijam {f}")
        if not self.files:
            raise RuntimeError("Nie znaleziono żadnych kompletnych par obraz-maska!")
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        image = cv2.imread(os.path.join(self.image_dir, img_name), cv2.IMREAD_GRAYSCALE)
        points = np.load(os.path.join(self.landmark_dir, img_name.replace('.png', '.npy')), allow_pickle=True)
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

# --- MODEL MULTISCALE HYBRIDGNET ---
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.theta = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.g     = nn.Conv2d(in_channels, in_channels//2, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels//2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        b = x.size(0)
        theta = self.theta(x).view(b, self.in_channels//2, -1).permute(0,2,1)
        phi   = self.phi(x).view(b, self.in_channels//2, -1)
        sim   = torch.matmul(theta, phi) / (self.in_channels//2)**0.5
        attn  = torch.softmax(sim, dim=-1)
        g     = self.g(x).view(b, self.in_channels//2, -1).permute(0,2,1)
        y     = torch.matmul(attn, g).permute(0,2,1).contiguous().view(b, self.in_channels//2, *x.shape[2:])
        y     = self.out_conv(y)
        y     = self.bn(y)
        return x + y

class MultiScaleHybridGNet(nn.Module):
    def __init__(self):
        super().__init__()
        def conv_block(in_ch, out_ch, dropout=0.1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            )
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.non_local = NonLocalBlock(256)
        self.middle    = conv_block(256, 512)
        self.up3       = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3      = conv_block(704, 256)
        self.up2       = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2      = conv_block(320, 128)
        self.up1       = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1      = conv_block(128, 64)
        self.final     = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m  = self.non_local(self.pool(e3))
        m  = self.middle(m)
        d3 = self.up3(m)
        ts = d3.shape[2:]
        e2r = F.interpolate(e2, size=ts, mode='bilinear', align_corners=False)
        e1r = F.interpolate(e1, size=ts, mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3, e2r, e1r], dim=1))
        d2 = self.up2(d3)
        ts = d2.shape[2:]
        e1r2 = F.interpolate(e1, size=ts, mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2, e1r2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)

# --- FUNKCJA STRATY ---
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, inputs, targets):
        inputs_sig = torch.sigmoid(inputs)
        inter = (inputs_sig * targets).sum()
        dice_l = 1 - (2 * inter) / (inputs_sig.sum() + targets.sum() + 1e-6)
        return self.bce(inputs, targets) + dice_l

# --- PRZYGOTOWANIE ZBIORU TESTOWEGO ---
full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR)
_, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- ZAŁADOWANIE MODELU ---
model = MultiScaleHybridGNet().to(DEVICE)
model.load_state_dict(torch.load("models/mshybridgnet_model50eb4lr4drop1.pth", map_location=DEVICE))
model.eval()

# --- TESTOWANIE Z METRYKAMI ---
criterion = DiceBCELoss()
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
        test_acc  += acc
        test_dice += dice
        test_iou  += iou
        preds_np  = preds.cpu().numpy()
        masks_np  = masks.cpu().numpy()
        for i in range(preds_np.shape[0]):
            hd   = hausdorff_distance(preds_np[i,0], masks_np[i,0])
            hd95v= hd95(preds_np[i,0], masks_np[i,0])
            if not np.isnan(hd):    sum_hd   += hd
            if not np.isnan(hd95v): sum_hd95 += hd95v
            count += 1

# --- WYNIKI ---
batches = len(test_loader)
print(f"Loss: {test_loss/batches:.4f}")
print(f"Accuracy: {test_acc/batches:.4f}")
print(f"Dice: {test_dice/batches:.4f}")
print(f"IoU: {test_iou/batches:.4f}")
if count > 0:
    print(f"Hausdorff Distance (avg): {sum_hd/count:.4f}")
    print(f"Hausdorff95 (avg): {sum_hd95/count:.4f}")

# --- WIZUALIZACJA ---
def plot_results(images, masks, preds, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples*5))
    for i in range(num_samples):
        axes[i,0].imshow(images[i,0].cpu().numpy(), cmap='gray'); axes[i,0].set_title("Obraz wejściowy"); axes[i,0].axis('off')
        axes[i,1].imshow(masks[i,0].cpu().numpy(), cmap='gray'); axes[i,1].set_title("Maska prawdziwa"); axes[i,1].axis('off')
        axes[i,2].imshow(preds[i,0].cpu().numpy(), cmap='gray'); axes[i,2].set_title("Predykcja modelu"); axes[i,2].axis('off')
    plt.tight_layout(); plt.show()

images, masks = next(iter(test_loader))
images, masks = images.to(DEVICE), masks.to(DEVICE)
with torch.no_grad():
    outputs = model(images)
    preds = (torch.sigmoid(outputs) > 0.5).float()
plot_results(images, masks, preds)