import yaml
import cv2
import numpy as np
import random
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
import albumentations as A
from tqdm import tqdm

def load_params(params_path="params.yaml"):
    """Loads project parameters from the YAML configuration file."""
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

# --- CUSTOM DATA LOADING ---

def load_custom_labeled_data(json_path, image_dir):
    """
    Integrates additional 500 labeled images into the pipeline.
    
    This function reads the COCO-formatted JSON, maps annotations to images,
    and returns a list of dictionaries that mimics the Hugging Face dataset structure.
    """
    if not Path(json_path).exists():
        print(f"Warning: Custom JSON not found at {json_path}. Skipping custom data.")
        return []

    with open(json_path, 'r') as f:
        coco = json.load(f)
    
    custom_items = []
    # Create a mapping of image IDs to their respective segmentation polygons
    img_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann['segmentation'])

    for img in coco['images']:
        img_path = Path(image_dir) / img['file_name']
        if not img_path.exists():
            continue
            
        # Read image to memory to standardize it alongside HF data
        image_np = cv2.imread(str(img_path))
        if image_np is None: continue
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Structure this to match 'item' format used in save_split()
        item = {
            "image": image_np,
            "segmentation": img_to_anns.get(img['id'], []),
            "custom": True 
        }
        custom_items.append(item)
    
    print(f"Successfully loaded {len(custom_items)} labelled images.")
    return custom_items

# --- GEOMETRIC CONVERSIONS ---

def polygons_to_mask(polygons, height, width):
    """Converts COCO-style polygon lists into a binary 1/0 numpy mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    if not polygons: return mask
    for poly in polygons:
        # COCO can nest polygons as [[x,y...]] or [x,y...]
        curr_poly_list = poly if isinstance(poly[0], list) else [poly]
        for sub_poly in curr_poly_list:
            points = np.array(sub_poly).flatten()
            if len(points) < 6: continue # Need at least 3 points (x,y)
            try:
                points = points.reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(mask, [points], color=1)
            except ValueError: continue
    return mask

def mask_to_yolo_polygons(mask, normalize=True):
    """Converts a binary mask into YOLO-formatted polygon strings (class 0)."""
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        # Filter out noise (very small areas)
        if cv2.contourArea(cnt) < 20: continue
        # Simplify the polygon to reduce file size without losing shape
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
    """Ensures all images (HF and Custom) are resized and cropped to the exact same size."""
    return A.Compose([
        A.SmallestMaxSize(max_size=img_size),
        A.CenterCrop(height=img_size, width=img_size),
    ])

# --- DATA PROCESSING PIPELINE ---

def save_split(dataset, split_name, output_dir, img_size):
    """Processes a data split (train/val/test) and saves images/labels to disk."""
    images_dir = output_dir / split_name / "images"
    labels_dir = output_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    transform_pipeline = get_structural_transforms(img_size)
    
    print(f"Standardizing {split_name} split...")
    for idx, item in enumerate(tqdm(dataset)):
        # 1. Normalize Image Format
        image = np.array(item["image"])
        if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # 2. Extract Polygons from metadata
        raw_polygons = []
        if "segmentation" in item: raw_polygons = item["segmentation"]
        elif "objects" in item and "segmentation" in item["objects"]: 
            raw_polygons = item["objects"]["segmentation"]
        
        h, w = image.shape[:2]
        mask = polygons_to_mask(raw_polygons, h, w)
            
        # 3. Apply Resize/Crop
        transformed = transform_pipeline(image=image, mask=mask)
        
        # 4. Save Image (JPG) and Label (TXT)
        filename = f"{split_name}_{idx:05d}"
        cv2.imwrite(str(images_dir / f"{filename}.jpg"), cv2.cvtColor(transformed["image"], cv2.COLOR_RGB2BGR))
        
        yolo_polys = mask_to_yolo_polygons(transformed["mask"])
        with open(labels_dir / f"{filename}.txt", "w") as f:
            if yolo_polys: f.write("\n".join(yolo_polys))

def main():
    params = load_params()
    seed = params["base"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    
    processed_dir = Path.cwd() / params["data"]["processed_dir"]
    img_size = params["data"]["img_size"]
    subset_ratio = params["data"]["subset"]

    # 0. Delete the processed folder
    if processed_dir.exists():
        print(f"Cleaning existing directory: {processed_dir}")
        shutil.rmtree(processed_dir) # Deletes the folder and everything in it
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Public Dataset from Hugging Face
    print("Loading HF dataset...")
    ds = load_dataset(params["data"]["hf_dataset"], name="full", trust_remote_code=True)
    full_ds = concatenate_datasets([ds['train'], ds['validation'], ds['test']])
    
    # Optional: Take a smaller subset for faster prototyping
    if subset_ratio < 1.0:
        full_ds = full_ds.shuffle(seed=seed).select(range(int(len(full_ds) * subset_ratio)))

    # 2. Split Public Data into Train, Val, and Test sets
    train_size = int(len(full_ds) * params["data"]["train_ratio"])
    val_size = int(len(full_ds) * params["data"]["val_ratio"])
    
    full_ds = full_ds.shuffle(seed=seed)
    train_ds = list(full_ds.select(range(0, train_size))) # List conversion allows .extend()
    val_ds = full_ds.select(range(train_size, train_size + val_size))
    test_ds = full_ds.select(range(train_size + val_size, len(full_ds)))

    # 3. Load and Merge Custom Data into the Training Set
    # We only add our custom data to 'train' to avoid data leakage into validation.
    custom_json = "./data/labeled/result.json"
    custom_img_dir = "./data/labeled/images"
    
    custom_train_data = load_custom_labeled_data(custom_json, custom_img_dir)
    train_ds.extend(custom_train_data)
    random.shuffle(train_ds) # Shuffle so custom data is distributed across batches

    # 4. Process all splits to YOLO format
    save_split(train_ds, "train", processed_dir, img_size)
    save_split(val_ds, "val", processed_dir, img_size)
    save_split(test_ds, "test", processed_dir, img_size)
    
    # 5. Generate YOLO data.yaml configuration
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
    
    print(f"\nPreprocessing Complete. Dataset saved to: {processed_dir}")

if __name__ == "__main__":
    main()