import os
import requests

BASE_URL = "https://api.isic-archive.com/api/v2/images"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "masks")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

mask_files = os.listdir(MASKS_DIR)

for mask_file in mask_files:
    if not mask_file.endswith("_segmentation.png"):
        continue

    image_id = mask_file.replace("_segmentation.png", "")
    print(f"Baixando imagem: {image_id}")

    response = requests.get(f"{BASE_URL}/{image_id}/")
    response.raise_for_status()

    data = response.json()
    image_url = data["files"]["full"]["url"]

    image_bytes = requests.get(image_url).content

    image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

print("Download das imagens finalizado!")
