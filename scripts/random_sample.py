from pathlib import Path
from enum import Enum
import json
import random
import shutil
from typing import Optional


TOTAL_SAMPLES = 5
SEED = 42
PARSED_LABELS_PATH = Path("visualization/parsed_labels.json")

class Object(Enum):
    TRAFFIC_LIGHT = "traffic light"
    LANE_ROAD_CURB = "lane/road curb"
    TRAFFIC_SIGN = "traffic sign"
    CAR = "car"
    AREA_DRIVABLE = "area/drivable"
    AREA_ALTERNATIVE = "area/alternative"
    LANE_SINGLE_WHITE = "lane/single white"
    PERSON = "person"
    LANE_DOUBLE_YELLOW = "lane/double yellow"
    BUS = "bus"
    LANE_SINGLE_YELLOW = "lane/single yellow"
    TRUCK = "truck"
    LANE_CROSSWALK = "lane/crosswalk"
    BIKE = "bike"
    RIDER = "rider"
    LANE_DOUBLE_WHITE = "lane/double white"
    MOTOR = "motor"
    TRAIN = "train"
    LANE_SINGLE_OTHER = "lane/single other"
    LANE_DOUBLE_OTHER = "lane/double other"
    AREA_UNKNOWN = "area/unknown"

def _load_parsed_labels():
    # Load parsed labels from parsed_labels.json file
    try:
        return json.loads(PARSED_LABELS_PATH.read_text())
    except Exception as e:
        raise FileNotFoundError(f"Could not load {PARSED_LABELS_PATH}: {e}. Please run parse_labels.py first.")

def make_dir(base_dir):
    # Delete sample directory if it exists before creating a new one with sub directories
    # Directories made: (sample, sample/images, sample/labels)
    base_dir = Path(base_dir).resolve(strict=True)
    sample_dir = base_dir / "sample"

    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    sample_dir.mkdir()
    (sample_dir / "images").mkdir()
    (sample_dir / "labels").mkdir()

def sample_dataset(base_dir, total_samples=TOTAL_SAMPLES, seed=SEED, object: Optional[Object] = None):
    # Sample from full dataset or based on object presence
    base_dir = Path(base_dir).resolve(strict=True)
    images_src = base_dir / "images"
    labels_src = base_dir / "labels"
    sample_dir = base_dir / "sample"
    images_dir = sample_dir / "images"
    labels_dir = sample_dir / "labels"

    rand = random.Random(seed)

    # Sample dataset from base_dir into sample directory
    if object is None:
        # Sample randomly from the full dataset
        candidates = list(labels_src.glob("*.json"))

        if total_samples > len(candidates):
            raise ValueError("total_samples cannot be greater than 100000 for the full dataset.")
        samples = rand.sample(candidates, total_samples)

        missing_images = []
        for label_path in samples:
            shutil.copy2(label_path, labels_dir / label_path.name)
            image_path = images_src / f"{label_path.stem}.jpg"
            if image_path.exists():
                shutil.copy2(image_path, images_dir / image_path.name)
            else:
                missing_images.append(label_path.name)

        return {"selected_count": len(samples), "missing_images_for": missing_images}
    else:
        # Sample based on object presence
        pass


def main():
    # Make sample directory and sub directories
    make_dir("data")
    # Sample from the full 100k dataset
    sample_dataset("data")

if __name__ == "__main__":
    main()