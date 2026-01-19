import os
import cv2
import numpy as np


# =========================
# BLACK-HAT MULTI-ORIENTATION
# =========================

def create_oriented_kernels(length=15, angles=(0, 30, 60, 90, 120, 150)):
    kernels = []
    for angle in angles:
        kernel = np.zeros((length, length), dtype=np.uint8)
        cv2.line(
            kernel,
            (length // 2, 0),
            (length // 2, length - 1),
            1,
            thickness=1
        )
        M = cv2.getRotationMatrix2D(
            (length // 2, length // 2),
            angle,
            1
        )
        kernel = cv2.warpAffine(kernel, M, (length, length))
        kernels.append(kernel)
    return kernels


def remove_hairs_blackhat(image_bgr, kernel_size=15, threshold=10):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernels = create_oriented_kernels(length=kernel_size)

    blackhat_responses = []
    for kernel in kernels:
        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        blackhat_responses.append(bh)

    blackhat_combined = np.max(np.stack(blackhat_responses, axis=0), axis=0)

    blackhat_combined = cv2.normalize(
        blackhat_combined, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)

    _, hair_mask = cv2.threshold(
        blackhat_combined, threshold, 255, cv2.THRESH_BINARY
    )

    hair_mask = cv2.dilate(
        hair_mask, np.ones((3, 3), np.uint8), iterations=2
    )

    image_no_hair = cv2.inpaint(
        image_bgr, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
    )

    return image_no_hair, hair_mask


# =========================
# PIPELINE
# =========================

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    images_dir = os.path.join(project_root, "data", "raw", "images")
    output_dir = os.path.join(project_root, "data", "processed", "no_hair")
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(".jpg")
    ])

    print(f"Processing {len(image_files)} images...")

    for idx, image_name in enumerate(image_files, start=1):
        print(f"[{idx}/{len(image_files)}] {image_name}")

        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"⚠️ Could not read {image_name}, skipping.")
            continue

        image_no_hair, _ = remove_hairs_blackhat(image)

        output_path = os.path.join(output_dir, image_name)
        cv2.imwrite(output_path, image_no_hair)

    print("✅ Hair removal completed for all images.")


if __name__ == "__main__":
    main()
