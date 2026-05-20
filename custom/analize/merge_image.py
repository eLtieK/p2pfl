import os
from pathlib import Path
from PIL import Image

# ======================
# CONFIG
# ======================
BASE_DIR = Path("custom/analize/images")  # sửa lại nếu cần
IMG_SIZE = (500, 500)  # resize cho đồng đều
COLUMNS = 3  # số ảnh mỗi hàng

# ======================
# PROCESS EACH FOLDER
# ======================
for folder in sorted(BASE_DIR.iterdir()):
    if not folder.is_dir():
        continue

    images = []
    for img_path in sorted(folder.glob("*.png")):
        if img_path.name == "merged.png":
            continue
        if img_path.name == "mean_loss.png":
            continue
        if img_path.name == "nodes_loss.png":
            continue
        if img_path.name == "compare_loss.png":
            continue
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            images.append(img)
        except:
            print(f"Skip {img_path}")

    if len(images) == 0:
        print(f"[SKIP] {folder} (no images)")
        continue

    # ======================
    # CREATE GRID
    # ======================
    rows = (len(images) + COLUMNS - 1) // COLUMNS

    grid_w = COLUMNS * IMG_SIZE[0]
    grid_h = rows * IMG_SIZE[1]

    grid_img = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, img in enumerate(images):
        x = (idx % COLUMNS) * IMG_SIZE[0]
        y = (idx // COLUMNS) * IMG_SIZE[1]
        grid_img.paste(img, (x, y))

    # ======================
    # SAVE INTO SAME FOLDER
    # ======================
    out_path = folder / "merged.png"
    grid_img.save(out_path)

    print(f"[DONE] {folder} -> merged.png")