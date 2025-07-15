import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from medpy.metric.binary import hd, hd95
import matplotlib.pyplot as plt

# --------------------
# 1. Dataset
# --------------------
class HeartDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        """
        root_dir: katalog główny z danymi,
                  wewnątrz powinny być podkatalogi 'Images' i 'landmarks'
        img_size: rozmiar, do którego będą skalowane obrazy i maski
        """
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'Images', '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'landmarks', '*.npy')))
        self.img_size = img_size
        
        # Transformacja: skalowanie + normalizacja dla obrazów
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Wczytanie obrazu w trybie RGB i transformacja
        img_path = self.image_paths[idx]
        img_pil = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_pil.size  # oryginalny rozmiar obrazu
        image = self.image_transform(img_pil)
        
        # Wczytanie punktów konturu z pliku .npy
        mask_path = self.mask_paths[idx]
        points = np.load(mask_path, allow_pickle=True)
        
        # Tworzenie maski na oryginalnym rozmiarze
        mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
        if points.size > 0:
            cv2.fillPoly(mask_np, [points.astype(np.int32)], color=255)
        # Konwersja do PIL i zmiana rozmiaru z NEAREST
        mask_img = Image.fromarray(mask_np)
        mask_img = mask_img.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = transforms.ToTensor()(mask_img)  # [1, H, W], wartości w [0,1]
        
        return image, mask

# --------------------
# 2. Metryki
# --------------------
def calculate_metrics(pred, target):
    """
    pred, target: [batch_size, 1, H, W]
    Zwraca: (accuracy, dice, iou, hd_mean, hd95_mean)
    """
    # Zakładamy, że pred są już prawdopodobieństwami (po Sigmoid), więc progowanie bez dodatkowej sigmoidy
    pred_bin = (pred > 0.5).float()
    target = target.float()
    
    # Podstawowe metryki
    tp = (pred_bin * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Hausdorff i HD95 (dla każdej próbki w batchu)
    hd_values = []
    hd95_values = []
    for i in range(pred_bin.shape[0]):
        pred_mask = pred_bin[i, 0].cpu().numpy().astype(bool)
        true_mask = target[i, 0].cpu().numpy().astype(bool)
        try:
            hd_val = hd(pred_mask, true_mask)
            hd95_val = hd95(pred_mask, true_mask)
        except:
            # Może się zdarzyć błąd, gdy maska pusta
            hd_val = 0
            hd95_val = 0
        hd_values.append(hd_val)
        hd95_values.append(hd95_val)
        
    return accuracy, dice, iou, np.mean(hd_values), np.mean(hd95_values)

# --------------------
# 3. Główna funkcja
# --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hiperparametry
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    img_size = 256
    
    # Przygotowanie datasetu
    dataset = HeartDataset("Chest-xray-landmark-dataset", img_size)
    train_size = int(0.8 * len(dataset))  # 80% na trening
    val_size = len(dataset) - train_size  # 20% na walidację
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # --------------------
    # 3.1. Wczytanie U-Net
    # --------------------
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch',
        'unet',
        in_channels=3,
        out_channels=1,
        init_features=32,
        pretrained=True  # lub False, jeśli nie chcemy wag pretrained
    )
    model = model.to(device)
    
    # --------------------
    # 3.2. Funkcja straty i optymalizator
    # --------------------
    criterion = nn.BCELoss()  # zamiast BCEWithLogitsLoss, bo model zwraca prawdopodobieństwa:contentReference[oaicite:1]{index=1}
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler obniży learning rate, jeśli val_loss nie poprawi się przez 3 epoki
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_dice = 0.0
    
    # Listy do zapisu metryk z epok
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    val_dice_list = []
    val_iou_list = []
    val_hd_list = []
    val_hd95_list = []
    
    # --------------------
    # 3.3. Pętla treningowa
    # --------------------
    for epoch in range(num_epochs):
        # --- Faza treningu ---
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # --- Faza walidacji ---
        model.eval()
        val_loss = 0.0
        metrics = {'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []}
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss_val = criterion(outputs, masks)
                val_loss += loss_val.item()
                
                acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
                metrics['acc'].append(acc.item())
                metrics['dice'].append(dice.item())
                metrics['iou'].append(iou.item())
                metrics['hd'].append(hd_val)
                metrics['hd95'].append(hd95_val)
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        
        # Średnie z metryk w tej epoce
        avg_acc = np.mean(metrics['acc'])
        avg_dice = np.mean(metrics['dice'])
        avg_iou = np.mean(metrics['iou'])
        avg_hd = np.mean(metrics['hd'])
        avg_hd95 = np.mean(metrics['hd95'])
        
        val_acc_list.append(avg_acc)
        val_dice_list.append(avg_dice)
        val_iou_list.append(avg_iou)
        val_hd_list.append(avg_hd)
        val_hd95_list.append(avg_hd95)
        
        # Redukcja LR, jeśli val_loss się nie poprawia
        scheduler.step(avg_val_loss)
        
        # Zapis najlepszego modelu (wg Dice)
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f"best_unet_dice_{avg_dice:.4f}.pth")
        
        # Wyświetlenie wyników
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {avg_acc:.4f}")
        print(f"Val Dice: {avg_dice:.4f}")
        print(f"Val IoU: {avg_iou:.4f}")
        print(f"Val HD: {avg_hd:.4f}")
        print(f"Val HD95: {avg_hd95:.4f}")
        print("----------------------------------")
    
    # --------------------
    # 3.4. Rysowanie wykresów
    # --------------------
    epochs = range(1, num_epochs + 1)
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    axs = axs.flatten()

    # Wykres 1: Train Loss
    axs[0].plot(epochs, train_loss_list, marker='o', label='Train Loss')
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Wykres 2: Validation Loss
    axs[1].plot(epochs, val_loss_list, marker='o', color='orange', label='Val Loss')
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    # Wykres 3: Validation Accuracy
    axs[2].plot(epochs, val_acc_list, marker='o', color='green', label='Val Accuracy')
    axs[2].set_title("Validation Accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].legend()

    # Wykres 4: Validation Dice
    axs[3].plot(epochs, val_dice_list, marker='o', color='red', label='Val Dice')
    axs[3].set_title("Validation Dice")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Dice")
    axs[3].legend()

    # Wykres 5: Validation IoU
    axs[4].plot(epochs, val_iou_list, marker='o', color='purple', label='Val IoU')
    axs[4].set_title("Validation IoU")
    axs[4].set_xlabel("Epoch")
    axs[4].set_ylabel("IoU")
    axs[4].legend()

    # Wykres 6: Validation Hausdorff Distance
    axs[5].plot(epochs, val_hd_list, marker='o', color='brown', label='Val HD')
    axs[5].set_title("Validation Hausdorff Distance")
    axs[5].set_xlabel("Epoch")
    axs[5].set_ylabel("HD")
    axs[5].legend()

    # Wykres 7: Validation HD95
    axs[6].plot(epochs, val_hd95_list, marker='o', color='magenta', label='Val HD95')
    axs[6].set_title("Validation HD95")
    axs[6].set_xlabel("Epoch")
    axs[6].set_ylabel("HD95")
    axs[6].legend()

    # Ostatni subplot niewykorzystany
    axs[7].axis('off')
    
    plt.tight_layout()
    plt.savefig("unet_pytorch_metrics.png")
    print("Wykres metryk zapisany jako 'unet_pytorch_metrics.png'")
    
    print("Training completed!")

# --------------------
# 4. Uruchomienie
# --------------------
if __name__ == "__main__":
    main()
