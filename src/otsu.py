import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from medpy.metric.binary import hd, hd95

# --- 1. WCZYTANIE OBRAZU ORAZ POPRAWA KONTRASTU (CLAHE) ---
image_path = 'Chest-xray-landmark-dataset/Images/1256842362861431725328351539259305635_u1qifz.png'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise IOError("Nie udało się wczytać obrazu.")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)

# --- 2. WYCIĘCIE ROI I SEGMENTACJA ---
# Dla uproszczenia definiujemy ROI jako stały fragment obrazu (można też użyć landmarków)
height, width = img_clahe.shape
roi_top = int(height * 0.3)
roi_bottom = int(height * 0.75)
roi_left = int(width * 0.3)
roi_right = int(width * 0.75)
roi = img_clahe[roi_top:roi_bottom, roi_left:roi_right]

# Segmentacja w ROI – np. metoda Otsu + operacje morfologiczne
_, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
kernel = np.ones((3, 3), np.uint8)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=2)
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
heart_mask_roi = np.zeros_like(roi)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(heart_mask_roi, [largest_contour], -1, 255, thickness=cv2.FILLED)

# Tworzymy pełnowymiarową maskę predykowaną, umieszczając wynik z ROI w oryginalnym obrazie:
pred_mask_full = np.zeros_like(img_clahe)
pred_mask_full[roi_top:roi_bottom, roi_left:roi_right] = heart_mask_roi
seg_mask_bin = (pred_mask_full > 127).astype(np.uint8)

# --- 3. WCZYTANIE MASKI GT (ORYGINALNEJ) ---
gt_landmarks_path = 'Chest-xray-landmark-dataset/landmarks/1256842362861431725328351539259305635_u1qifz.npy'
landmarks = np.load(gt_landmarks_path, allow_pickle=True)
gt_mask = np.zeros_like(img, dtype=np.uint8)
if landmarks.size > 0:
    points = landmarks.astype(np.int32)
    cv2.fillPoly(gt_mask, [points], color=1)

# --- 4. ROZSZERZENIE MASKI GT DO ROZMIARÓW MASKI PREDYKOWANEJ ---
# Oblicz bounding box dla maski predykowanej:
pred_coords = np.column_stack(np.where(seg_mask_bin > 0))
if pred_coords.size == 0:
    raise ValueError("Maska predykowana jest pusta!")
pred_y_min, pred_x_min = pred_coords.min(axis=0)
pred_y_max, pred_x_max = pred_coords.max(axis=0)
pred_box_width = pred_x_max - pred_x_min
pred_box_height = pred_y_max - pred_y_min

# Oblicz bounding box dla oryginalnych landmarków (maski GT pierwotnej)
land_coords = np.column_stack(np.where(gt_mask > 0))
if land_coords.size == 0:
    raise ValueError("Maska GT jest pusta!")
land_y_min, land_x_min = land_coords.min(axis=0)
land_y_max, land_x_max = land_coords.max(axis=0)
land_box_width = land_x_max - land_x_min
land_box_height = land_y_max - land_y_min

# Wytnij oryginalną maskę GT z obszaru landmarków:
gt_crop = gt_mask[land_y_min:land_y_max, land_x_min:land_x_max]

# Przeskaluj wyciętą maskę GT do rozmiaru bounding boxa maski predykowanej:
resized_gt_crop = cv2.resize(gt_crop, (pred_box_width, pred_box_height), interpolation=cv2.INTER_NEAREST)

# Utwórz nową maskę GT o rozmiarach oryginalnego obrazu (wszystkie zera) i wklej przeskalowany fragment w miejsce bounding boxa predykcji:
expanded_gt_mask = np.zeros_like(gt_mask)
expanded_gt_mask[pred_y_min:pred_y_max, pred_x_min:pred_x_max] = resized_gt_crop

# --- 5. OBLICZENIE METRYK ---
import numpy as np
from sklearn.metrics import accuracy_score
from medpy.metric.binary import hd, hd95

# zamieniamy na boolowskie maski
pred = seg_mask_bin.astype(bool)
gt   = expanded_gt_mask.astype(bool)

# True positive i suma pikseli
intersection = np.logical_and(pred, gt).sum()
sum_pred_gt  = pred.sum() + gt.sum()
union        = np.logical_or(pred, gt).sum()

# Dice: 2·|A∩B|/(|A|+|B|)
dice = (2 * intersection) / sum_pred_gt if sum_pred_gt > 0 else 1.0

# IoU: |A∩B|/|A∪B|
iou = intersection / union if union > 0 else 1.0

# Accuracy: od razu na spłaszczonych integerach
accuracy = accuracy_score(gt.astype(np.uint8).ravel(),
                          pred.astype(np.uint8).ravel())

# HD i HD95 z obroną pustych masek
if pred.any() and gt.any():
    hd_val   = hd   (pred.astype(np.uint8), gt.astype(np.uint8))
    hd95_val = hd95 (pred.astype(np.uint8), gt.astype(np.uint8))
else:
    # gdy co najmniej jedna maska jest pusta
    hd_val, hd95_val = np.inf, np.inf

print(f"Accuracy: {accuracy:.4f}")
print(f"IoU:      {iou:.4f}")
print(f"Dice:     {dice:.4f}")
print(f"HD:       {hd_val:.2f}")
print(f"HD95:     {hd95_val:.2f}")

# --- 6. WIZUALIZACJA ---
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Obraz oryginalny")
ax[0].axis('off')
ax[1].imshow(expanded_gt_mask, cmap='gray')
ax[1].set_title("Maska GT rozszerzona")
ax[1].axis('off')
ax[2].imshow(seg_mask_bin, cmap='gray')
ax[2].set_title("Maska predykowana")
ax[2].axis('off')
plt.tight_layout()
plt.show()
