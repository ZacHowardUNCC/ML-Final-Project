import json
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog


def preprocess_images(selected_object, downscale_factor, images_dir="data/sample/images", labels_dir="data/sample/labels"):
    object_name = selected_object.value
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    X = []
    y = []

    for image_path in sorted(images_dir.glob("*.jpg")):
        label_path = labels_dir / f"{image_path.stem}.json"

        if not label_path.exists():
            print(f"[WARN] Missing label for {label_path}, skipping.")
            continue

        label = json.loads(label_path.read_text())
        has_object = any(objects["category"] == object_name for objects in label.get("frames",[{}])[0].get("objects", []))

        # Load and preprocess image
        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[WARN] Unable to read image {image_path}, skipping.")
            continue

        height, width = image.shape[:2]
        image = cv2.resize(image, (width // downscale_factor, height // downscale_factor))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Extract HOG features
        hog_features = hog(
            image_gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys', # Normalization method
            transform_sqrt=True,
            visualize=False,
            feature_vector=True
        )

        X.append(hog_features)
        y.append(1 if has_object else 0)

    X = np.array(X)
    y = np.array(y)
    np.savez("data/preprocessed_data.npz", X=X, y=y)
    print(f"Preprocessed {len(X)} images. Positive samples: {np.sum(y)}, Negative samples: {len(y) - np.sum(y)}")

def main():
    print("dont run this script directly")

if __name__ == "__main__":
    main()