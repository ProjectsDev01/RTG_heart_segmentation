import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
from medpy.metric.binary import hd, hd95

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DEFINICJA TRANSFORMACJI DLA OBRAZÓW ---
img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- DEFINICJA DATASETU ---
class HeartDataset(Dataset):
    def __init__(self, image_dir, landmark_dir, transform=None):
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        self.files = [f for f in sorted(os.listdir(image_dir)) if f.endswith('.png') and \
                      os.path.exists(os.path.join(landmark_dir, f.replace('.png', '.npy')))]
        if not self.files:
            raise RuntimeError("Nie znaleziono par obraz-maska!")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Wczytanie obrazu
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Nie znaleziono obrazu: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Tworzenie maski z punktów konturu
        pts = np.load(os.path.join(self.landmark_dir, img_name.replace('.png', '.npy')), allow_pickle=True)
        mask_np = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
        if pts.size > 0:
            cv2.fillPoly(mask_np, [pts.astype(np.int32)], color=255)

        # Transformacja obrazu
        image = self.transform(img_rgb) if self.transform else transforms.ToTensor()(img_rgb)

        # Skalowanie maski i konwersja na tensor [1, H, W]
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize(IMG_SIZE, Image.NEAREST)
        mask = transforms.ToTensor()(mask_pil)

        return image, mask

# --- METRYKI ---
def calculate_metrics(pred, target):
    pred_bin = (pred > 0.5).float()
    target_f = target.float()
    tp = (pred_bin * target_f).sum()
    fp = (pred_bin * (1 - target_f)).sum()
    fn = ((1 - pred_bin) * target_f).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)

    hd_vals = []
    hd95_vals = []
    for i in range(pred_bin.shape[0]):
        p = pred_bin[i, 0].cpu().numpy().astype(bool)
        t = target_f[i, 0].cpu().numpy().astype(bool)
        try:
            hd_vals.append(hd(p, t))
            hd95_vals.append(hd95(p, t))
        except:
            hd_vals.append(0)
            hd95_vals.append(0)
    return (precision.item(), recall.item(), dice.item(), iou.item(),
            float(np.mean(hd_vals)), float(np.mean(hd95_vals)))

# --- ŁADOWANIE MODELU ---
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=False)
model = model.to(DEVICE)
model_state = torch.load("models/unet_dice_0.8524.pth", map_location=DEVICE)
if next(iter(model_state)).startswith('module.'):
    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
model.load_state_dict(model_state)
model.eval()

# --- PRZYGOTOWANIE DANYCH ---
full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR, transform=img_transform)
_, test_idx = train_test_split(list(range(len(full_dataset))), test_size=0.2, random_state=42)
test_loader = DataLoader(Subset(full_dataset, test_idx), batch_size=BATCH_SIZE, shuffle=False)

# --- TESTOWANIE ---
criterion = nn.BCELoss()
test_metrics = {'loss': 0, 'precision': 0, 'recall': 0, 'dice': 0, 'iou': 0, 'hd': 0, 'hd95': 0}
num_batches = len(test_loader)
with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testowanie"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)

        loss = criterion(outputs, masks)
        p, r, d, i, h_dist, h95 = calculate_metrics(outputs, masks)
        test_metrics['loss'] += loss.item()
        test_metrics['precision'] += p
        test_metrics['recall'] += r
        test_metrics['dice'] += d
        test_metrics['iou'] += i
        test_metrics['hd'] += h_dist
        test_metrics['hd95'] += h95

print("\nWyniki testowe U-Net:")
print(f"Loss: {test_metrics['loss']/num_batches:.4f}")
print(f"Precision: {test_metrics['precision']/num_batches:.4f}")
print(f"Recall: {test_metrics['recall']/num_batches:.4f}")
print(f"Dice: {test_metrics['dice']/num_batches:.4f}")
print(f"IoU: {test_metrics['iou']/num_batches:.4f}")
print(f"HD: {test_metrics['hd']/num_batches:.4f}")
print(f"HD95: {test_metrics['hd95']/num_batches:.4f}")

# --- WIZUALIZACJA ---
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor.cpu() * std + mean

def plot_predictions(num_samples=3):
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(test_loader))
        images = images.to(DEVICE)
        outputs = model(images)
        preds = (outputs > 0.5)

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        for i in range(num_samples):
            img = denormalize(images[i]).permute(1, 2, 0).numpy()
            axes[i, 0].imshow(np.clip(img, 0, 1))
            axes[i, 0].axis('off'); axes[i, 0].set_title('Obraz wejściowy')
            axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 1].axis('off'); axes[i, 1].set_title('Maska GT')
            axes[i, 2].imshow(preds[i, 0].cpu().numpy(), cmap='gray')
            axes[i, 2].axis('off'); axes[i, 2].set_title('Predykcja')
        plt.tight_layout()
        plt.show()

plot_predictions()
