import os
import glob
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from medpy.metric.binary import hd, hd95
import matplotlib.pyplot as plt

# --------------------
# Dataset
# --------------------
class HeartDataset(Dataset):
    def __init__(self, root_dir, img_size=256):
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'Images', '*.png')))
        self.mask_paths = sorted(glob.glob(os.path.join(root_dir, 'landmarks', '*.npy')))
        self.img_size = img_size
        # Transform dla obrazu
        self.image_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Wczytanie obrazu jako grayscale, przeskalowanie i konwersja do RGB
        img_path = self.image_paths[idx]
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise RuntimeError(f"Nie udało się wczytać obrazu: {img_path}")
        img_resized = cv2.resize(img_gray, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img = self.image_transform(img_pil)

        # Tworzenie maski z punktów
        points = np.load(self.mask_paths[idx], allow_pickle=True)
        # Ładujemy oryginalny rozmiar maski
        orig = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask_full = np.zeros(orig.shape, dtype=np.uint8)
        cv2.fillPoly(mask_full, [points.astype(np.int32)], color=1)
        mask_resized = cv2.resize(mask_full, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask_resized).unsqueeze(0).float()

        return img, mask

# --------------------
# Metryki
# --------------------
def calculate_metrics(pred, target):
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    target = target.float()

    tp = (pred_bin * target).sum()
    tn = ((1 - pred_bin) * (1 - target)).sum()
    fp = (pred_bin * (1 - target)).sum()
    fn = ((1 - pred_bin) * target).sum()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    hd_vals, hd95_vals = [], []
    for i in range(pred_bin.shape[0]):
        pm = pred_bin[i,0].cpu().numpy().astype(bool)
        tm = target[i,0].cpu().numpy().astype(bool)
        try:
            hd_vals.append(hd(pm, tm))
            hd95_vals.append(hd95(pm, tm))
        except:
            hd_vals.append(0.0)
            hd95_vals.append(0.0)
    return accuracy.item(), dice.item(), iou.item(), np.mean(hd_vals), np.mean(hd95_vals)

# --------------------
# Model i funkcje pomocnicze
# --------------------
def create_deeplabv3(num_classes=1, pretrained=True):
    model = models.segmentation.deeplabv3_resnet50(pretrained=pretrained)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model

def freeze_batchnorm_layers(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d): m.eval()

# --------------------
# Główny skrypt
# --------------------
def main():
    checkpoint_path = 'models/resnet50.pth'
    data_root = 'Chest-xray-landmark-dataset'
    img_size = 256
    batch_size = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    dataset = HeartDataset(data_root, img_size)
    test_size = int(0.2 * len(dataset))
    _, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_size, test_size])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = create_deeplabv3(num_classes=1, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    filtered = {k:v for k,v in state.items() if not k.startswith('aux_classifier.')}
    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    freeze_batchnorm_layers(model)

    accs, dices, ious, hds, hd95s = [], [], [], [], []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)['out']
            a, d, i, h, h95 = calculate_metrics(out, masks)
            accs.append(a); dices.append(d); ious.append(i); hds.append(h); hd95s.append(h95)
    print("Test Results:")
    print(f"Accuracy: {np.mean(accs):.4f}")
    print(f"Dice:     {np.mean(dices):.4f}")
    print(f"IoU:      {np.mean(ious):.4f}")
    print(f"HD:       {np.mean(hds):.4f}")
    print(f"HD95:     {np.mean(hd95s):.4f}")

    # --------------------
    # Wizualizacja
    # --------------------
    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad(): preds = (torch.sigmoid(model(imgs)['out']) > 0.5).float()

    # Pierwszy przykład
    img = imgs[0].cpu().numpy()
    # Denormalizacja
    mean = np.array([0.485,0.456,0.406]).reshape(3,1,1)
    std = np.array([0.229,0.224,0.225]).reshape(3,1,1)
    img = img * std + mean
    img = np.transpose(img, (1,2,0))
    img = np.clip(img, 0, 1)

    gt = masks[0,0].cpu().numpy()
    pr = preds[0,0].cpu().numpy()

    fig, ax = plt.subplots(1,3,figsize=(12,4))
    ax[0].imshow(img); ax[0].set_title('Image'); ax[0].axis('off')
    ax[1].imshow(gt, cmap='gray'); ax[1].set_title('GT Mask'); ax[1].axis('off')
    ax[2].imshow(pr, cmap='gray'); ax[2].set_title('Pred Mask'); ax[2].axis('off')
    plt.tight_layout(); plt.savefig('example_prediction.png')
    print("Example saved as example_prediction.png")

if __name__=='__main__':
    main()
