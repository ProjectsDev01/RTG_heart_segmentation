import cv2
import matplotlib.pyplot as plt
import numpy as np
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

# Zapamiętanie oryginalnych wymiarów
IMG_HEIGHT, IMG_WIDTH = xray_image.shape
print("Wymiary RTG:", (IMG_HEIGHT, IMG_WIDTH))

normalized_image = cv2.normalize(xray_image, None, 0, 255, cv2.NORM_MINMAX)

def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    equalized = cv2.equalizeHist(blurred)
    return equalized

# -------------------------
# 2. Detekcja ROI serca (obszar do segmentacji)
# -------------------------
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

template = preprocess_image(normalized_image)
region = detect_heart_region(template)
if region is None:
    raise ValueError("Nie udało się wyznaczyć regionu bazowego.")
x1, y1, x2, y2 = region

# Poszerzenie ROI o margines
base_top    = max(0, y1-20)
base_bottom = min(IMG_HEIGHT, y2+20)
base_left   = max(0, x1-20)
base_right  = min(IMG_WIDTH, x2+20)
base_region = template[base_top:base_bottom, base_left:base_right].copy()
h_roi, w_roi = base_region.shape

# CLAHE w ROI
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
base_clahe = clahe.apply(base_region)

# -------------------------
# 3. Przygotowanie masek „pewnych” do Watershed
# -------------------------
# 3.1 Progowanie Otsu na base_clahe
_, bin_mask = cv2.threshold(base_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
bin_mask = cv2.morphologyEx(
    bin_mask,
    cv2.MORPH_CLOSE,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)),
    iterations=2
)

# 3.2 Wyznaczenie sure_bg (tło) przez dylatację
kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
sure_bg = cv2.dilate(bin_mask, kernel_bg, iterations=3)

# 3.3 Nowe sure_fg: używamy elipsy wydłużonej w poziomie
cx, cy = w_roi // 2, h_roi // 2
ellipse_mask = np.zeros_like(base_clahe, dtype=np.uint8)
# elipsa: 90% szerokości ROI, 50% wysokości ROI (wydłużony poziomo)
cv2.ellipse(
    ellipse_mask,
    (cx, cy),
    (int(w_roi * 0.45), int(h_roi * 0.25)),
    0, 0, 360,
    255,
    -1
)

# sure_fg = fragmenty bin_mask * w elipsie
sure_fg = cv2.bitwise_and(bin_mask, bin_mask, mask=ellipse_mask)
# Erozja, by zostały tylko pewne obszary centralne
kernel_fg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
sure_fg = cv2.erode(sure_fg, kernel_fg, iterations=2)
sure_fg = np.uint8(sure_fg)

# 3.4 Obszar „unknown” = sure_bg - sure_fg
unknown = cv2.subtract(sure_bg, sure_fg)

# -------------------------
# 4. Inicjalizacja markerów i Watershed
# -------------------------
# 4.1 Tworzymy tablicę markerów: tło=1, przód>1, unknown=0
markers = np.zeros((h_roi, w_roi), dtype=np.int32)
markers[sure_bg == 0] = 1
num_fg, fg_labels = cv2.connectedComponents(sure_fg)
markers[fg_labels > 0] = fg_labels[fg_labels > 0] + 1

# 4.2 Gradient (morfologiczny) jako teren
grad = cv2.morphologyEx(
    base_clahe,
    cv2.MORPH_GRADIENT,
    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
)

# 4.3 Aplikacja Watershed
roi_color = cv2.cvtColor(base_clahe, cv2.COLOR_GRAY2BGR)
cv2.watershed(roi_color, markers)

# -------------------------
# 5. Budowa finalnej maski serca w ROI i postprocessing
# -------------------------
mask = np.zeros_like(base_clahe, dtype=np.uint8)
mask[markers > 1] = 255

# Morfologia: domknięcie + otwarcie
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2, iterations=2)

# Wybór największego konturu i ograniczenie do 70% ROI
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if cnts:
    c = max(cnts, key=cv2.contourArea)
    mask[:] = 0
    cv2.drawContours(mask, [c], -1, 255, -1)
    area_c = cv2.contourArea(c)
    maxA = 0.7 * mask.size
    if area_c > maxA:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx_c, cy_c = M['m10']/M['m00'], M['m01']/M['m00']
            sf = math.sqrt(maxA / area_c)
            pts = np.array(
                [
                    [[int(cx_c + (p[0][0]-cx_c)*sf), int(cy_c + (p[0][1]-cy_c)*sf)]]
                    for p in c
                ],
                dtype=np.int32
            )
            mask[:] = 0
            cv2.fillPoly(mask, [pts], 255)

# -------------------------
# 6. Integracja do pełnego obrazu
# -------------------------
pred_full = np.zeros_like(normalized_image, dtype=np.uint8)
pred_full[base_top:base_bottom, base_left:base_right] = (mask > 127).astype(np.uint8)

# -------------------------
# 7. Wczytanie i skalowanie maski GT z landmarków
# -------------------------
gt_path = 'Chest-xray-landmark-dataset/landmarks/1256842362861431725328351539259305635_u1qifz.npy'
gt_pts = np.load(gt_path, allow_pickle=True)
if gt_pts.size == 0:
    raise ValueError("Puste landmarki w pliku GT")

ORIG_SIZE = (1024, 1024)
scale_x = IMG_WIDTH / ORIG_SIZE[1]
scale_y = IMG_HEIGHT / ORIG_SIZE[0]

gt_pts_scaled = gt_pts.astype(np.float32)
gt_pts_scaled[:, 0] *= scale_x
gt_pts_scaled[:, 1] *= scale_y

gt_full = np.zeros_like(xray_image, dtype=np.uint8)
cv2.fillPoly(gt_full, [gt_pts_scaled.astype(np.int32)], 1)

# -------------------------
# 8. Metryki: Accuracy, IoU, Dice, HD, HD95
# -------------------------
p = pred_full.astype(bool)
g = gt_full.astype(bool)

intersection = np.logical_and(p, g).sum()
union        = np.logical_or(p, g).sum()

dice_val = (2 * intersection) / (p.sum() + g.sum()) if (p.sum()+g.sum()) > 0 else 1.0
iou_val  = intersection / union if union > 0 else 1.0
acc_val  = accuracy_score(g.ravel().astype(int), p.ravel().astype(int))

if p.any() and g.any():
    hd_val   = hd(p.astype(np.uint8), g.astype(np.uint8))
    hd95_val = hd95(p.astype(np.uint8), g.astype(np.uint8))
else:
    hd_val, hd95_val = np.inf, np.inf

print(f"Accuracy: {acc_val:.4f}")
print(f"IoU:      {iou_val:.4f}")
print(f"Dice:     {dice_val:.4f}")
print(f"HD:       {hd_val:.2f}")
print(f"HD95:     {hd95_val:.2f}")

# -------------------------
# 9. Wizualizacja
# -------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(normalized_image, cmap='gray')
axs[0].set_title('Obraz RTG')
axs[0].axis('off')

axs[1].imshow(gt_full, cmap='gray')
axs[1].set_title('Maska GT (po skalowaniu)')
axs[1].axis('off')

axs[2].imshow(pred_full, cmap='gray')
axs[2].set_title('Predykcja Watershed')
axs[2].axis('off')

plt.tight_layout()
plt.show()
