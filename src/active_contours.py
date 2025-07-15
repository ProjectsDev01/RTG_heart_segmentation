import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import morphological_chan_vese
import math
from sklearn.metrics import accuracy_score
from medpy.metric.binary import hd, hd95

# -------------------------
# 1. Wczytanie obrazu, normalizacja i preprocessing
# -------------------------
image_path = 'Chest-xray-landmark-dataset/Images/1256842362861431725328351539259305635_u1qifz.png'
xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if xray_image is None:
    raise IOError("Nie udało się wczytać obrazu!")
normalized_image = cv2.normalize(xray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized


def detect_heart_region(image):
    h, w = image.shape
    top, bottom = int(h*0.3), int(h*0.7)
    left, right = int(w*0.3), int(w*0.8)
    roi = image[top:bottom, left:right]
    _, thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(c)
    return left+x, top+y, left+x+rw, top+y+rh

# 1.5. Przygotowanie ROI bazowego
template = preprocess_image(normalized_image)
region = detect_heart_region(template)
if region is None:
    raise ValueError("Nie udało się wyznaczyć regionu bazowego.")
x1, y1, x2, y2 = region
base_top = max(0, y1-20); base_bottom = min(template.shape[0], y2+20)
base_left = max(0, x1-20); base_right = min(template.shape[1], x2+20)
base_region = template[base_top:base_bottom, base_left:base_right]

# CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
base_clahe = clahe.apply(base_region)

# -------------------------
# 2. Wybór seed point
# -------------------------
hb, wb = base_clahe.shape
rt, rb = int(hb*0.35), int(hb*0.65)
lt, lr = int(wb*0.35), int(wb*0.65)
roi_small = base_clahe[rt:rb, lt:lr]
_, _, _, maxLoc = cv2.minMaxLoc(roi_small)
seed = (maxLoc[0]+lt, maxLoc[1]+rt)

# -------------------------
# 3. Inicjalizacja koła
# -------------------------
area = base_region.size
r = math.sqrt(0.5*area/math.pi)
s = np.linspace(0, 2*math.pi, 200)
circle = np.vstack([seed[0]+r*np.cos(s), seed[1]+r*np.sin(s)]).T

# -------------------------
# 4. Poziom inicjalny (level set)
# -------------------------
init_ls = np.zeros_like(base_clahe, dtype=np.int8)
cv2.circle(init_ls, seed, int(r), 1, -1)

# -------------------------
# 5. Segmentacja + morfologia
# -------------------------
mask = morphological_chan_vese(base_clahe, num_iter=150, init_level_set=init_ls, smoothing=3)
mask = (mask.astype(np.uint8))*255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(mask)
    cv2.drawContours(mask, [c], -1, 255, -1)
# ograniczenie rozlewania
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    c = max(cnts, key=cv2.contourArea)
    area_c = cv2.contourArea(c)
    maxA = 0.7*base_region.size
    if area_c>maxA:
        M = cv2.moments(c)
        if M['m00']!=0:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
            sf = math.sqrt(maxA/area_c)
            pts = [[[int(cx+(pt[0][0]-cx)*sf), int(cy+(pt[0][1]-cy)*sf)]] for pt in c]
            c = np.array(pts, dtype=np.int32)
            m2 = np.zeros_like(mask)
            cv2.fillPoly(m2, [c], 255)
            mask = m2

# -------------------------
# 6. Pełna maska predykcyjna
# -------------------------
pred_full = np.zeros_like(normalized_image, dtype=np.uint8)
pred_full[base_top:base_bottom, base_left:base_right] = (mask>127).astype(np.uint8)

# -------------------------
# 7. Maska GT
# -------------------------
gt_path = 'Chest-xray-landmark-dataset/landmarks/1256842362861431725328351539259305635_u1qifz.npy'
gt_pts = np.load(gt_path, allow_pickle=True)
gt_full = np.zeros_like(xray_image, dtype=np.uint8)
if gt_pts.size>0:
    cv2.fillPoly(gt_full, [gt_pts.astype(np.int32)], 1)
else:
    raise ValueError("Brak GT landmarks")

# -------------------------
# 8. Rozszerzenie GT + metryki
# -------------------------
# bbox pred
coords_p = np.column_stack(np.where(pred_full>0))
y0p, x0p = coords_p.min(axis=0); y1p, x1p = coords_p.max(axis=0)
# bbox gt
coords_g = np.column_stack(np.where(gt_full>0))
y0g, x0g = coords_g.min(axis=0); y1g, x1g = coords_g.max(axis=0)
# crop & resize
crop = gt_full[y0g:y1g, x0g:x1g]
hp, wp = y1p-y0p, x1p-x0p
gt_resized = cv2.resize(crop, (wp, hp), interpolation=cv2.INTER_NEAREST)
# expanded
expanded_gt = np.zeros_like(gt_full, dtype=np.uint8)
expanded_gt[y0p:y1p, x0p:x1p] = gt_resized

# bool masks
p = pred_full.astype(bool)
g = expanded_gt.astype(bool)
# inter/union
inter = np.logical_and(p,g).sum()
uni   = np.logical_or(p,g).sum()
# dice
dice = (2*inter)/(p.sum()+g.sum()) if (p.sum()+g.sum())>0 else 1.0
# iou
iou  = inter/uni if uni>0 else 1.0
# acc
acc  = accuracy_score(g.ravel().astype(int), p.ravel().astype(int))
# hd & hd95
if p.any() and g.any():
    hdv   = hd(p.astype(np.uint8), g.astype(np.uint8))
    hd95v = hd95(p.astype(np.uint8), g.astype(np.uint8))
else:
    hdv, hd95v = np.inf, np.inf

print(f"Accuracy: {acc:.4f}")
print(f"IoU:      {iou:.4f}")
print(f"Dice:     {dice:.4f}")
print(f"HD:       {hdv:.2f}")
print(f"HD95:     {hd95v:.2f}")

# -------------------------
# 9. Wizualizacja
# -------------------------
fig, axs = plt.subplots(1,3,figsize=(15,5))
axs[0].imshow(normalized_image, cmap='gray'); axs[0].set_title('Obraz'); axs[0].axis('off')
axs[1].imshow(expanded_gt, cmap='gray'); axs[1].set_title('GT'); axs[1].axis('off')
axs[2].imshow(pred_full, cmap='gray'); axs[2].set_title('Predykcja'); axs[2].axis('off')
plt.tight_layout()
plt.show()
