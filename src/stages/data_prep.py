import yaml
import cv2
import numpy as np
import random
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import albumentations as A
from tqdm import tqdm

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons: return mask
    for poly in polygons:
        points = np.array(poly).flatten()
        if len(points) < 6: continue
        try:
            points = points.reshape(-1, 2).astype(np.int32)
            cv2.fillPoly(mask, [points], color=1)
        except ValueError: continue
    return mask

def mask_to_yolo_polygons(mask, normalize=True):
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 20: continue
        cnt = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
        if len(cnt) < 3: continue
        coords = cnt.flatten().astype(float)
        if normalize:
            coords[0::2] /= w
            coords[1::2] /= h
        coords = np.clip(coords, 0, 1)
        poly_str = "0 " + " ".join([f"{x:.6f}" for x in coords])
        polygons.append(poly_str)
    return polygons

def get_structural_transforms(img_size):
    """
    ONLY Structural changes. No random flips/rotations here.
    1. Resize so smallest side matches target.
    2. Center crop to target.
    This guarantees every output is a valid 500x500 square.
    """
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size),
        A.CenterCrop(height=img_size, width=img_size),
    ])

def save_split(dataset, split_name, output_dir, img_size):
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # We use the same pipeline for Train, Val, and Test now
    # because this is just standardization, not augmentation.
    transform_pipeline = get_structural_transforms(img_size)
    
    print(f"Standardizing {split_name} ({len(dataset)} images)...")
    for idx, item in enumerate(tqdm(dataset)):
        # Load & Fix Color
        image = np.array(item["image"])
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Extract Mask
        raw_polygons = []
        if "objects" in item and "segmentation" in item["objects"]: raw_polygons = item["objects"]["segmentation"]
        elif "segmentation" in item: raw_polygons = item["segmentation"]
        
        h, w = image.shape[:2]
        mask = polygons_to_mask(raw_polygons, h, w)
            
        # Apply Structure (Crop/Resize)
        augmented = transform_pipeline(image=image, mask=mask)
        
        # Save
        filename = f"{split_name}_{idx:05d}"
        cv2.imwrite(str(images_dir / f"{filename}.jpg"), cv2.cvtColor(augmented["image"], cv2.COLOR_RGB2BGR))
        
        polygons = mask_to_yolo_polygons(augmented["mask"])
        with open(labels_dir / f"{filename}.txt", "w") as f:
            if polygons: f.write("\n".join(polygons))

def main():
    params = load_params()
    seed = params["base"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    
    processed_dir = Path.cwd() / params["data"]["processed_dir"]
    img_size = params["data"]["img_size"]
    subset_ratio = params["data"]["subset"]
    
    print("Loading dataset...")
    ds = load_dataset(params["data"]["hf_dataset"], name="full", trust_remote_code=True)
    full_ds = concatenate_datasets([ds['train'], ds['validation'], ds['test']])
    
    if subset_ratio < 1.0:
        full_ds = full_ds.shuffle(seed=seed).select(range(int(len(full_ds) * subset_ratio)))

    # Splits
    train_size = int(len(full_ds) * params["data"]["train_ratio"])
    val_size = int(len(full_ds) * params["data"]["val_ratio"])
    
    full_ds = full_ds.shuffle(seed=seed)
    train_ds = full_ds.select(range(0, train_size))
    val_ds = full_ds.select(range(train_size, train_size + val_size))
    test_ds = full_ds.select(range(train_size + val_size, len(full_ds)))
    
    # Process All Splits (Same Logic)
    save_split(train_ds, "train", processed_dir, img_size)
    save_split(val_ds, "val", processed_dir, img_size)
    save_split(test_ds, "test", processed_dir, img_size)
    
    # Generate data.yaml
    yaml_content = f"""
path: {processed_dir.absolute()} 
train: train/images
val: val/images
test: test/images
names:
  0: building
"""
    with open(processed_dir / "data.yaml", "w") as f:
        f.write(yaml_content)

if __name__ == "__main__":
    main()