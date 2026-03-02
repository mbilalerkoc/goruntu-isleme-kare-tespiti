import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import yaml

# ==========================
# AYARLAR
# ==========================
IMG_SIZE = 1024
NUM_IMAGES = 1200
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1

BASE_PATH = "iha_realistic_dataset"
CLASSES = {
    0: "blue_square_4x4",
    1: "red_square_2x2"
}

# ==========================
# KLASÖR OLUŞTUR
# ==========================
for split in ["train", "val", "test"]:
    os.makedirs(f"{BASE_PATH}/images/{split}", exist_ok=True)
    os.makedirs(f"{BASE_PATH}/labels/{split}", exist_ok=True)

# ==========================
# GERÇEKÇİ ASFALT TEXTURE
# ==========================
def generate_asphalt(size):
    base = np.random.normal(110, 25, (size, size)).astype(np.uint8)
    texture = cv2.GaussianBlur(base, (9,9), 0)
    asphalt = cv2.merge([texture, texture, texture])
    return asphalt

# ==========================
# DRONE PERSPEKTİFİ
# ==========================
def perspective_transform(img):
    h, w = img.shape[:2]
    margin = random.randint(100, 250)

    pts1 = np.float32([
        [margin, margin],
        [w-margin, margin],
        [margin, h-margin],
        [w-margin, h-margin]
    ])

    pts2 = np.float32([
        [random.randint(0, 200), random.randint(0, 200)],
        [w-random.randint(0,200), random.randint(0,200)],
        [random.randint(0,200), h-random.randint(0,200)],
        [w-random.randint(0,200), h-random.randint(0,200)]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (w, h))
    return warped

# ==========================
# YOLO LABEL FORMAT
# ==========================
def yolo_format(x, y, w, h):
    return f"{x} {y} {w} {h}"

# ==========================
# ANA ÜRETİM
# ==========================
for i in tqdm(range(NUM_IMAGES)):

    img = generate_asphalt(IMG_SIZE)
    label_lines = []

    # 4x4 MAVİ
    size_blue = random.randint(180, 300)
    x_blue = random.randint(0, IMG_SIZE - size_blue)
    y_blue = random.randint(0, IMG_SIZE - size_blue)

    cv2.rectangle(img,
                  (x_blue, y_blue),
                  (x_blue + size_blue, y_blue + size_blue),
                  (255,0,0), -1)

    xc = (x_blue + size_blue/2) / IMG_SIZE
    yc = (y_blue + size_blue/2) / IMG_SIZE
    w = size_blue / IMG_SIZE
    h = size_blue / IMG_SIZE

    label_lines.append(f"0 {xc} {yc} {w} {h}")

    # 2x2 KIRMIZI
    size_red = random.randint(90, 160)
    x_red = random.randint(0, IMG_SIZE - size_red)
    y_red = random.randint(0, IMG_SIZE - size_red)

    cv2.rectangle(img,
                  (x_red, y_red),
                  (x_red + size_red, y_red + size_red),
                  (0,0,255), -1)

    xc = (x_red + size_red/2) / IMG_SIZE
    yc = (y_red + size_red/2) / IMG_SIZE
    w = size_red / IMG_SIZE
    h = size_red / IMG_SIZE

    label_lines.append(f"1 {xc} {yc} {w} {h}")

    # Distractor shapes (label yok)
    cv2.circle(img,
               (random.randint(0,IMG_SIZE), random.randint(0,IMG_SIZE)),
               random.randint(30,80),
               (random.randint(0,255),random.randint(0,255),random.randint(0,255)),
               -1)

    # Drone perspektifi
    img = perspective_transform(img)

    # Motion blur
    if random.random() < 0.3:
        ksize = random.choice([5,7,9])
        img = cv2.GaussianBlur(img, (ksize,ksize), 0)

    # Split
    if i < NUM_IMAGES * TRAIN_SPLIT:
        split = "train"
    elif i < NUM_IMAGES * (TRAIN_SPLIT + VAL_SPLIT):
        split = "val"
    else:
        split = "test"

    cv2.imwrite(f"{BASE_PATH}/images/{split}/img_{i}.jpg", img)

    with open(f"{BASE_PATH}/labels/{split}/img_{i}.txt", "w") as f:
        f.write("\n".join(label_lines))

# data.yaml
with open(f"{BASE_PATH}/data.yaml", "w") as f:
    yaml.dump({
        "path": BASE_PATH,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASSES
    }, f)

print("DATASET TAMAMLANDI")