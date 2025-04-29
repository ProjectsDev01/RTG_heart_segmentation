import torch
import torch.nn as nn
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
                npy = f.replace('.png', '.npy')
                if os.path.exists(os.path.join(landmark_dir, npy)):
                    self.files.append(f)
        if not self.files:
            raise RuntimeError("Nie znaleziono żadnych kompletnych par obraz-maska!")
        self.files.sort()

    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        f = self.files[idx]
        img = cv2.imread(os.path.join(self.image_dir, f), cv2.IMREAD_GRAYSCALE)
        points = np.load(os.path.join(self.landmark_dir, f.replace('.png', '.npy')), allow_pickle=True)
        mask = np.zeros(img.shape, dtype=np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        img = cv2.resize(img, IMG_SIZE) / 255.0
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
        return (torch.tensor(img, dtype=torch.float32).unsqueeze(0),
                torch.tensor(mask, dtype=torch.float32).unsqueeze(0))

# --- METRYKI DICE, IOU, ACC ---
def calculate_seg_metrics(pred, target):
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

# --- MODEL U-NET ---
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def dbl(in_ch, out_ch): return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )
        self.enc1, self.enc2, self.enc3 = dbl(1,64), dbl(64,128), dbl(128,256)
        self.pool = nn.MaxPool2d(2)
        self.middle = dbl(256,512)
        self.up3 = nn.ConvTranspose2d(512,256,2,2); self.dec3 = dbl(512,256)
        self.up2 = nn.ConvTranspose2d(256,128,2,2); self.dec2 = dbl(256,128)
        self.up1 = nn.ConvTranspose2d(128,64,2,2);  self.dec1 = dbl(128,64)
        self.final = nn.Conv2d(64,1,1)
    def forward(self, x):
        e1 = self.enc1(x);
        e2 = self.enc2(self.pool(e1));
        e3 = self.enc3(self.pool(e2));
        m  = self.middle(self.pool(e3));
        d3 = self.dec3(torch.cat([self.up3(m), e3],1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2],1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1],1))
        return self.final(d1)

# --- STRATA ---
class DiceBCELoss(nn.Module):
    def __init__(self): super().__init__(); self.bce = nn.BCEWithLogitsLoss()
    def forward(self, inp, tgt):
        inp_sig = torch.sigmoid(inp)
        inter = (inp_sig * tgt).sum()
        dice_l = 1 - (2*inter + 1e-6) / (inp_sig.sum() + tgt.sum() + 1e-6)
        return self.bce(inp, tgt) + dice_l

# --- PRZYGOTOWANIE ZBIORU ---
full_ds = HeartDataset(IMG_DIR, LANDMARK_DIR)
_, test_idx = train_test_split(list(range(len(full_ds))), test_size=0.2, random_state=42)
test_ds = Subset(full_ds, test_idx)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- ZAŁADOWANIE MODELU ---
model = UNet().to(DEVICE)
model.load_state_dict(torch.load("models/unet_model50eb4lr4drop1.pth", map_location=DEVICE, weights_only=True))
model.eval()

# --- TESTOWANIE I METRYKI ---
crit = DiceBCELoss()
tot_loss = tot_acc = tot_dice = tot_iou = 0.0
cnt = 0
sum_hd = sum_hd95 = 0.0
with torch.no_grad():
    for imgs, masks in tqdm(test_ld, desc="Testowanie"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        outs = model(imgs)
        loss = crit(outs, masks)
        acc, dice, iou, preds = calculate_seg_metrics(outs, masks)
        tot_loss += loss.item()
        tot_acc += acc; tot_dice += dice; tot_iou += iou
        # HD metrics per sample
        preds_np = preds.cpu().numpy()
        targs_np = masks.cpu().numpy()
        for i in range(preds_np.shape[0]):
            hd_val = hausdorff_distance(preds_np[i,0], targs_np[i,0])
            hd95_val = hd95(preds_np[i,0], targs_np[i,0])
            if not np.isnan(hd_val): sum_hd += hd_val
            if not np.isnan(hd95_val): sum_hd95 += hd95_val
            cnt += 1

# --- WYNIKI ---
batches = len(test_ld)
samples = cnt
print(f"Loss: {tot_loss/batches:.4f}")
print(f"Accuracy: {tot_acc/batches:.4f}")
print(f"Dice: {tot_dice/batches:.4f}")
print(f"IoU: {tot_iou/batches:.4f}")
print(f"Hausdorff Distance (avg): {sum_hd/samples:.4f}")
print(f"Hausdorff95 (avg): {sum_hd95/samples:.4f}")

# --- WIZUALIZACJA ---
def plot_results(imgs, masks, preds, n=3):
    fig, ax = plt.subplots(n,3,figsize=(15,n*5))
    for i in range(n):
        ax[i,0].imshow(imgs[i,0].cpu(),cmap='gray'); ax[i,0].set_title("Input"); ax[i,0].axis('off')
        ax[i,1].imshow(masks[i,0].cpu(),cmap='gray'); ax[i,1].set_title("Mask"); ax[i,1].axis('off')
        ax[i,2].imshow(preds[i,0].cpu(),cmap='gray'); ax[i,2].set_title("Pred"); ax[i,2].axis('off')
    plt.tight_layout(); plt.show()

imgs, masks = next(iter(test_ld))
imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
with torch.no_grad():
    outs = model(imgs)
    preds = (torch.sigmoid(outs)>0.5).float()
plot_results(imgs, masks, preds)
