import os
import cv2
import numpy as np
import re
from tqdm import tqdm
from PIL import Image

# Constants for skin detection and area filtering
lower_skin = np.array([0, 10, 60], dtype=np.uint8)
upper_skin = np.array([25, 180, 255], dtype=np.uint8)
MIN_HAND_AREA = 500
MAX_HAND_AREA = 200000

def preprocess_image(img):
    """Preprocess a single image to extract hand region."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    if not (MIN_HAND_AREA < area < MAX_HAND_AREA):
        return None

    x, y, w, h = cv2.boundingRect(max_contour)
    padding = int(max(w, h) * 0.2)
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    hand_region = img[y:y+h, x:x+w]
    hand_mask = mask[y:y+h, x:x+w]
    img_masked = cv2.bitwise_and(hand_region, hand_region, mask=hand_mask)
    # Ensure 3 channels
    if len(img_masked.shape) == 2:
        img_masked = cv2.cvtColor(img_masked, cv2.COLOR_GRAY2BGR)
    # Resize to (96, 96)
    img_masked = cv2.resize(img_masked, (96, 96))
    return img_masked

def preprocess_training_data(raw_data_dir, processed_data_dir):
    """Preprocess all training images to remove backgrounds."""
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    processed_count = 0
    skipped_count = 0

    for filename in tqdm(os.listdir(raw_data_dir), desc="Processing images"):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(raw_data_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            skipped_count += 1
            continue

        # Extract class name from filename (everything before last underscore + number)
        match = re.match(r'(.+)_\d+\.(jpg|jpeg|png)', filename, re.IGNORECASE)
        if not match:
            print(f"Could not extract class from filename: {filename}")
            skipped_count += 1
            continue
        class_name = match.group(1)

        output_filename = f"{class_name}_processed_{processed_count}.png"
        output_path = os.path.join(processed_data_dir, output_filename)

        processed_img = preprocess_image(img)
        if processed_img is not None:
            cv2.imwrite(output_path, processed_img)
            processed_count += 1
        else:
            print(f"Skipped (no hand detected or area out of range): {img_path}")
            skipped_count += 1

    print(f"\nProcessed {processed_count} images.")
    print(f"Skipped {skipped_count} images.")

    # Verify and remove corrupted images
    print("\nVerifying processed images...")
    corrupted = []
    for fname in os.listdir(processed_data_dir):
        fpath = os.path.join(processed_data_dir, fname)
        try:
            with Image.open(fpath) as img:
                img.verify()
        except Exception as e:
            print(f"Corrupted image: {fpath} ({e})")
            corrupted.append(fpath)
    for fpath in corrupted:
        try:
            os.remove(fpath)
            print(f"Removed corrupted image: {fpath}")
        except Exception as e:
            print(f"Failed to remove corrupted image: {fpath} ({e})")
    print(f"Verification complete. Removed {len(corrupted)} corrupted images.")

if __name__ == "__main__":
    raw_data_dir = r"E:\new yousef\hand-gesture-recognition\data\raw"
    processed_data_dir = r"E:\new yousef\hand-gesture-recognition\data\processed"
    preprocess_training_data(raw_data_dir, processed_data_dir)