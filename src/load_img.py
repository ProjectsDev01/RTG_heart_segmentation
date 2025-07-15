import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Ścieżki
DATA_DIR     = "Chest-xray-landmark-dataset"
IMG_DIR      = os.path.join(DATA_DIR, "Images")
LANDMARK_DIR = os.path.join(DATA_DIR, "landmarks")

basename = "1256842362861431725328351539259305635_u1qifz"

img_path  = os.path.join(IMG_DIR,      basename + ".png")
land_path = os.path.join(LANDMARK_DIR, basename + ".npy")

# Wczytanie obrazu
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise IOError(f"Nie udało się wczytać obrazu: {img_path}")

print("Wymiary RTG:", img.shape)  # (wysokość, szerokość)

# Wczytanie punktów
pts = np.load(land_path, allow_pickle=True)
if pts.size == 0:
    raise ValueError(f"Puste landmarki w pliku: {land_path}")

print("Pierwsze 5 punktów przed skalowaniem:", pts[:5])
print("Min punktów:", pts.min(axis=0))
print("Max punktów:", pts.max(axis=0))

# Założenie, że landmarki były robione na obrazie 1024x1024
ORIG_SIZE = (1024, 1024)  # (wysokość, szerokość)

# Skalowanie punktów do rozmiaru RTG
scale_x = img.shape[1] / ORIG_SIZE[1]
scale_y = img.shape[0] / ORIG_SIZE[0]

pts_scaled = np.copy(pts).astype(np.float32)
pts_scaled[:, 0] *= scale_x
pts_scaled[:, 1] *= scale_y

print("Pierwsze 5 punktów po skalowaniu:", pts_scaled[:5])
print("Min po skalowaniu:", pts_scaled.min(axis=0))
print("Max po skalowaniu:", pts_scaled.max(axis=0))

# Tworzenie maski
mask = np.zeros_like(img, dtype=np.uint8)
poly = pts_scaled.astype(np.int32).reshape(-1, 1, 2)
cv2.fillPoly(mask, [poly], 255)

# Wyświetlenie RTG i maski
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img, cmap='gray')
axes[0].set_title("Obraz RTG")
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title("Maska z przeskalowanych punktów")
axes[1].axis('off')

plt.tight_layout()
plt.show()
