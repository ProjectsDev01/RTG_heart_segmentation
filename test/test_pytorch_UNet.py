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

# --- KONFIGURACJA ---
DATA_DIR = "Chest-xray-landmark-dataset"
IMG_DIR = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")
IMG_SIZE = (256, 256)
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Definicja transformacji - analogiczna do treningowej
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
        
        # Wyszukaj kompletne pary obraz-maska
        self.files = []
        for f in os.listdir(image_dir):
            if f.endswith('.png'):
                landmark_name = f.replace('.png', '.npy')
                landmark_path = os.path.join(landmark_dir, landmark_name)
                if os.path.exists(landmark_path):
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
        # Wczytywanie obrazu jako kolorowy
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Nie udało się wczytać obrazu: {img_path}")
        # Konwersja BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Wczytywanie maski
        landmark_path = os.path.join(self.landmark_dir, img_name.replace('.png', '.npy'))
        points = np.load(landmark_path, allow_pickle=True)
        mask = np.zeros(image.shape[:2], dtype=np.float32)
        cv2.fillPoly(mask, [points.astype(np.int32)], color=1)
        
        # Przetwarzanie
        if self.transform:
            image = self.transform(image)
        else:
            # Alternatywnie: skalowanie i konwersja do tensora bez normalizacji
            image = cv2.resize(image, IMG_SIZE) / 255.0
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
            
        mask = cv2.resize(mask, IMG_SIZE, interpolation=cv2.INTER_NEAREST)
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

# --- ŁADOWANIE MODELU ---
# Używamy tego samego modelu co w treningu:
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                       in_channels=3, out_channels=1, init_features=32, pretrained=True)
model = model.to(DEVICE)
# Upewnij się, że ścieżka do wagi jest poprawna
model_state = torch.load("models/unet_dice_0.6969.pth", map_location=DEVICE)
model.load_state_dict(model_state, strict=False)
model.eval()

# --- PRZYGOTOWANIE ZBIORU TESTOWEGO ---
full_dataset = HeartDataset(IMG_DIR, LANDMARK_DIR, transform=img_transform)
indices = list(range(len(full_dataset)))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42)  # 20% danych testowych
test_dataset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- TESTOWANIE ---
test_loss = 0
test_acc = 0
test_dice = 0
test_iou = 0
criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
    for images, masks in tqdm(test_loader, desc="Testowanie"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)
        acc, dice, iou = calculate_metrics(outputs, masks)
        
        test_loss += loss.item()
        test_acc += acc
        test_dice += dice
        test_iou += iou

num_batches = len(test_loader)
print("\nWyniki testowe U-Net:")
print(f"Loss: {test_loss/num_batches:.4f}")
print(f"Accuracy: {test_acc/num_batches:.4f}")
print(f"Dice: {test_dice/num_batches:.4f}")
print(f"IoU: {test_iou/num_batches:.4f}")

# --- WIZUALIZACJA WYNIKÓW ---
def plot_results(images, masks, predictions, num_samples=3):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    for i in range(num_samples):
        # Obraz wejściowy (odwracamy normalizację dla wizualizacji)
        img = images[i].cpu().clone()
        # Odwrócenie normalizacji
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        img = img * std + mean
        img = img.clamp(0, 1)
        
        axes[i, 0].imshow(img.permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Obraz wejściowy")
        axes[i, 0].axis('off')
        
        # Maska referencyjna
        axes[i, 1].imshow(masks[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Maska prawdziwa")
        axes[i, 1].axis('off')
        
        # Predykcja modelu
        axes[i, 2].imshow(predictions[i, 0].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title("Predykcja modelu")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Pobierz przykładowe dane z loadera
images, masks = next(iter(test_loader))
images, masks = images.to(DEVICE), masks.to(DEVICE)

with torch.no_grad():
    outputs = model(images)
    predictions = (torch.sigmoid(outputs) > 0.5).float()

plot_results(images, masks, predictions)
