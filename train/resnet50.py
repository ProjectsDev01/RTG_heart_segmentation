import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm import tqdm
from medpy.metric.binary import hd, hd95
import matplotlib.pyplot as plt

# --------------------
# 1. Dataset
# --------------------
class HeartDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'Images', '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'landmarks', '*.npy')))
        self.img_size = img_size
        
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Wczytanie obrazu
        img = Image.open(self.image_paths[idx]).convert('RGB')
        img = self.image_transform(img)
        
        # Wczytanie maski
        mask_arr = np.load(self.mask_paths[idx])  # zakładamy, że jest to binaryzowana maska (0 i 1)
        mask_img = Image.fromarray(mask_arr.astype(np.uint8)).resize(
            (self.img_size, self.img_size), Image.NEAREST
        )
        mask = transforms.ToTensor()(mask_img)  # [1, H, W]

        return img, mask

# --------------------
# 2. Metryki
# --------------------
def calculate_metrics(pred, target):
    """
    pred i target to tensory [batch_size, 1, H, W]
    """
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()
    
    # Basic metrics
    tp = (pred_bin * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    
    # Hausdorff (dla każdej próbki w batchu)
    hd_values = []
    hd95_values = []
    for i in range(pred_bin.shape[0]):
        pred_mask = pred_bin[i, 0].cpu().numpy().astype(bool)
        true_mask = target[i, 0].cpu().numpy().astype(bool)
        try:
            hd_val = hd(pred_mask, true_mask)
            hd95_val = hd95(pred_mask, true_mask)
        except:
            hd_val = 0
            hd95_val = 0
        hd_values.append(hd_val)
        hd95_values.append(hd95_val)
        
    return accuracy, dice, iou, np.mean(hd_values), np.mean(hd95_values)

# --------------------
# 3. Funkcja pomocnicza: Tworzymy model
# --------------------
def create_deeplabv3(num_classes=1, pretrained=True):
    """
    Tworzy model Deeplabv3 z backbone ResNet50, z wczytanymi wagami 
    ImageNet (pretrained=True). Następnie modyfikujemy ostatnią warstwę 
    na 1 kanał (binarny problem).
    """
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    # Zamiast domyślnej klasyfikacji (21 kanałów dla VOC) -> 1 kanał
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def freeze_batchnorm_layers(model):
    """
    Ustawia wszystkie warstwy BatchNorm2d w modelu w tryb ewaluacji.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

# --------------------
# 4. Główny skrypt
# --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Hyperparameters
    num_epochs = 50
    batch_size = 4
    learning_rate = 1e-4
    img_size = 256
    
    # Dataset i DataLoaders
    dataset = HeartDataset("Chest-xray-landmark-dataset", img_size)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
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
    
    # Tworzymy model Deeplabv3 z 1 kanałem wyjściowym
    model = create_deeplabv3(num_classes=1, pretrained=True).to(device)
    
    # Zamrażamy warstwy BatchNorm, aby uniknąć błędu przy małej liczbie elementów na kanał
    freeze_batchnorm_layers(model)
    
    # Funkcja straty i optymalizator
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    best_dice = 0.0

    # Listy do zapisu metryk
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    val_dice_list = []
    val_iou_list = []
    val_hd_list = []
    val_hd95_list = []
    
    for epoch in range(num_epochs):
        # --------------------
        # 4.1. Training phase
        # --------------------
        model.train()
        # Upewnij się, że przed każdą epoką BatchNorm są w trybie ewaluacji
        freeze_batchnorm_layers(model)
        
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, masks in progress_bar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            # forward
            outputs = model(images)['out']  # Deeplabv3 zwraca słownik, bierzemy 'out'
            loss = criterion(outputs, masks)
            # backward
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # --------------------
        # 4.2. Validation phase
        # --------------------
        model.eval()
        val_loss = 0.0
        metrics = {'acc': [], 'dice': [], 'iou': [], 'hd': [], 'hd95': []}
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, masks in val_progress:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                acc, dice, iou, hd_val, hd95_val = calculate_metrics(outputs, masks)
                metrics['acc'].append(acc.item())
                metrics['dice'].append(dice.item())
                metrics['iou'].append(iou.item())
                metrics['hd'].append(hd_val)
                metrics['hd95'].append(hd95_val)
        
        avg_val_loss = val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        avg_dice = np.mean(metrics['dice'])
        val_dice_list.append(avg_dice)
        val_acc_list.append(np.mean(metrics['acc']))
        val_iou_list.append(np.mean(metrics['iou']))
        val_hd_list.append(np.mean(metrics['hd']))
        val_hd95_list.append(np.mean(metrics['hd95']))
        
        # Redukcja LR na podstawie val_loss
        scheduler.step(avg_val_loss)
        
        # Zapis najlepszego modelu (wg Dice)
        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), f"best_deeplab_dice_{avg_dice:.4f}.pth")
        
        # Wyświetlenie wyników epoki
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {np.mean(metrics['acc']):.4f}")
        print(f"Val Dice: {avg_dice:.4f}")
        print(f"Val IoU: {np.mean(metrics['iou']):.4f}")
        print(f"Val HD: {np.mean(metrics['hd']):.4f}")
        print(f"Val HD95: {np.mean(metrics['hd95']):.4f}")
        print("----------------------------------")
    
    # --------------------
    # 4.3. Rysowanie wykresów
    # --------------------
    epochs_range = range(1, num_epochs + 1)
    fig, axs = plt.subplots(4, 2, figsize=(12, 16))
    axs = axs.flatten()

    # Train Loss
    axs[0].plot(epochs_range, train_loss_list, marker='o', label='Train Loss')
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Val Loss
    axs[1].plot(epochs_range, val_loss_list, marker='o', color='orange', label='Val Loss')
    axs[1].set_title("Validation Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    # Val Accuracy
    axs[2].plot(epochs_range, val_acc_list, marker='o', color='green', label='Val Accuracy')
    axs[2].set_title("Validation Accuracy")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Accuracy")
    axs[2].legend()

    # Val Dice
    axs[3].plot(epochs_range, val_dice_list, marker='o', color='red', label='Val Dice')
    axs[3].set_title("Validation Dice")
    axs[3].set_xlabel("Epoch")
    axs[3].set_ylabel("Dice")
    axs[3].legend()

    # Val IoU
    axs[4].plot(epochs_range, val_iou_list, marker='o', color='purple', label='Val IoU')
    axs[4].set_title("Validation IoU")
    axs[4].set_xlabel("Epoch")
    axs[4].set_ylabel("IoU")
    axs[4].legend()

    # Val HD
    axs[5].plot(epochs_range, val_hd_list, marker='o', color='brown', label='Val HD')
    axs[5].set_title("Validation Hausdorff Distance")
    axs[5].set_xlabel("Epoch")
    axs[5].set_ylabel("HD")
    axs[5].legend()

    # Val HD95
    axs[6].plot(epochs_range, val_hd95_list, marker='o', color='magenta', label='Val HD95')
    axs[6].set_title("Validation HD95")
    axs[6].set_xlabel("Epoch")
    axs[6].set_ylabel("HD95")
    axs[6].legend()

    # Ostatni subplot pusty
    axs[7].axis('off')
    
    plt.tight_layout()
    plt.savefig("deeplab_pytorch_metrics.png")
    print("Wykres metryk zapisany jako 'deeplab_pytorch_metrics.png'")
    
    print("Training completed!")

if __name__ == "__main__":
    main()
