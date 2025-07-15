import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the X-ray image
image_path = 'Chest-xray-landmark-dataset/Images/1256842362861431725328351539259305635_u1qifz.png'
xray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Normalize the grayscale intensities
normalized_image = cv2.normalize(xray_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

# Function to preprocess the image
def preprocess_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Enhance contrast using histogram equalization
    equalized_image = cv2.equalizeHist(blurred_image)
    return equalized_image

# Function to detect the largest contour in the central area
def detect_heart_region(image):
    height, width = image.shape
    # Define the central region of interest
    central_top = int(height * 0.3)
    central_bottom = int(height * 0.7)
    central_left = int(width * 0.3)
    central_right = int(width * 0.8)
    central_region = image[central_top:central_bottom, central_left:central_right]
    # Apply binary thresholding
    _, binary_image = cv2.threshold(central_region, 100, 255, cv2.THRESH_BINARY)
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Get bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Adjust coordinates back to the original image
        x_min = central_left + x
        y_min = central_top + y
        x_max = x_min + w
        y_max = y_min + h
        return x_min, y_min, x_max, y_max
    return None

# Preprocess the image
preprocessed_image = preprocess_image(normalized_image)

# Detect the heart region
heart_region = detect_heart_region(preprocessed_image)

# Create the base region around the detected heart region if found
if heart_region:
    x_min, y_min, x_max, y_max = heart_region
    # Define the base region (dodajemy margines 20 pikseli)
    base_top = max(0, y_min - 20)
    base_bottom = min(preprocessed_image.shape[0], y_max + 20)
    base_left = max(0, x_min - 20)
    base_right = min(preprocessed_image.shape[1], x_max + 20)
    # Extract the base region
    base_region = preprocessed_image[base_top:base_bottom, base_left:base_right]
    # Detect heart region within the extracted base region
    base_contours, _ = cv2.findContours(base_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if base_contours:
        largest_base_contour = max(base_contours, key=cv2.contourArea)
        cv2.drawContours(base_region, [largest_base_contour], -1, (255, 255, 255), 2)
else:
    base_region = None

# Display results in subplots (układ 2x3)
fig, ax = plt.subplots(2, 3, figsize=(12, 10))
ax = ax.ravel()

# 1. Original image
ax[0].imshow(normalized_image, cmap='gray')
ax[0].set_title("Original Image")
ax[0].axis("off")

# 2. Preprocessed image
ax[1].imshow(preprocessed_image, cmap='gray')
ax[1].set_title("Preprocessed Image")
ax[1].axis("off")

# 3. Heart region and Base overlay
heart_overlay = cv2.cvtColor(normalized_image, cv2.COLOR_GRAY2BGR)
if heart_region:
    cv2.rectangle(heart_overlay, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Heart bounding box
    cv2.rectangle(heart_overlay, (base_left, base_top), (base_right, base_bottom), (255, 0, 0), 2)  # Base region
ax[2].imshow(cv2.cvtColor(heart_overlay, cv2.COLOR_BGR2RGB))
ax[2].set_title("Heart Region and Base")
ax[2].axis("off")

# 4. Extracted Heart Region (base_region)
if base_region is not None:
    ax[3].imshow(base_region, cmap='gray')
    ax[3].set_title("Extracted Heart Region")
    ax[3].axis("off")
else:
    ax[3].text(0.5, 0.5, "No base region", horizontalalignment='center', verticalalignment='center')
    ax[3].set_title("Extracted Heart Region")
    ax[3].axis("off")

# ---- Zwiększenie kontrastu w wyciętym obszarze za pomocą CLAHE ----
if base_region is not None:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    base_region_clahe = clahe.apply(base_region)
else:
    base_region_clahe = None

# 5. Obszar po zwiększeniu kontrastu (CLAHE)
if base_region_clahe is not None:
    ax[4].imshow(base_region_clahe, cmap='gray')
    ax[4].set_title("CLAHE Enhanced Region")
    ax[4].axis("off")
else:
    ax[4].text(0.5, 0.5, "No CLAHE region", horizontalalignment='center', verticalalignment='center')
    ax[4].set_title("CLAHE Enhanced Region")
    ax[4].axis("off")

# 6. Pusty subplot (lub dodatkowa wizualizacja, np. binaryzacja) - ukrywamy go
ax[5].axis("off")

plt.tight_layout()
plt.show()

# Generate binary mask for the "base" region
mask_base2 = np.zeros_like(preprocessed_image, dtype=np.uint8)
if heart_region:
    cv2.rectangle(mask_base2, (base_left, base_top), (base_right, base_bottom), 255, thickness=cv2.FILLED)
mask_base2_binary = (mask_base2 > 0).astype(np.uint8)
# cv2.imwrite('mask_base2.png', mask_base2_binary * 255)

# ---- Histogram wyciętego regionu po zwiększeniu kontrastu z osią x przedstawioną w percentylach ----
if base_region_clahe is not None:
    # Oblicz histogram (dla 256 przedziałów)
    hist, bins = np.histogram(base_region_clahe.ravel(), bins=256, range=(0,256))
    # Oblicz centra przedziałów
    bin_centers = (bins[:-1] + bins[1:]) / 2
    # Oblicz dystrybuantę (CDF) histogramu
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1] * 100  # przelicz na procenty
    # Interpolacja środka przedziału na odpowiadający percentyl
    percentiles = np.interp(bin_centers, bins[1:], cdf_normalized)
    
    # Wyświetlenie histogramu w osobnej figurze
    plt.figure(figsize=(6, 4))
    plt.plot(percentiles, hist, color='black')
    plt.title("Histogram of CLAHE Enhanced Region")
    plt.xlabel("Percentile")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
