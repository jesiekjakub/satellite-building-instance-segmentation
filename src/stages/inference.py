import torch
import cv2
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms as T
import random

# Import your architecture definitions
from src.models.custom_architecture import get_custom_model
from src.models.transfer_learning import get_transfer_model
# Import Dataset to load Ground Truth automatically
from src.utils.dataset import SatelliteDataset

def load_params(path="params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_model(params, device):
    """
    Loads the architecture and weights based on params.yaml configuration.
    """
    model_type = params["train"]["model_type"]
    weights_path = Path(f"runs/{model_type}_model/weights/best.pth")
    
    print(f"Loading {model_type.upper()} model configuration...")
    print(f"Weights source: {weights_path}")
    
    if model_type == "transfer":
        model = get_transfer_model(num_classes=2)
    else:
        model = get_custom_model(num_classes=2)
    
    model.to(device)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found at {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    return model

def generate_random_color():
    return [random.randint(50, 255) for _ in range(3)]

def visualize_instance_segmentation(image, masks, boxes, scores=None, threshold=0.5, title=""):
    """
    Draws masks and boxes on an image. 
    Works for both Predictions (with scores) and Ground Truth (no scores).
    """
    img_display = image.copy()
    
    # Handle case where masks/boxes are tensors
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().detach().numpy()
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().detach().numpy()
    if scores is not None and isinstance(scores, torch.Tensor):
        scores = scores.cpu().detach().numpy()
    
    # If no objects found
    if len(boxes) == 0:
        return img_display

    for i in range(len(boxes)):
        # If prediction, filter by score
        if scores is not None:
            if scores[i] < threshold: continue
            score_val = scores[i]
        else:
            score_val = 1.0 # Ground Truth always exists

        # 1. Unique Color
        color = generate_random_color()
        
        # 2. Draw Mask
        # Ground Truth masks are usually (H, W), Predictions are (1, H, W)
        mask = masks[i]
        if mask.ndim == 3: mask = mask[0] 
        
        mask_bool = mask > threshold
        if mask_bool.any():
            roi = img_display[mask_bool]
            alpha = 0.5
            blended = (roi.astype(float) * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
            img_display[mask_bool] = blended

        # 3. Draw Box
        box = boxes[i].astype(int)
        cv2.rectangle(img_display, (box[0], box[1]), (box[2], box[3]), color, 2)
        
        # 4. Label (Score for Pred, "GT" for Ground Truth)
        if scores is not None:
            label = f"{score_val:.2f}"
        else:
            label = "GT"
            
        cv2.putText(img_display, label, (box[0], box[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Add Title at top
    cv2.putText(img_display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img_display

def main():
    params = load_params()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup
    output_dir = Path("runs/inference_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Load Model
    model = get_model(params, device)
    
    # 3. Load Test Dataset (This gives us Images + Ground Truth Labels!)
    print("Loading Test Set...")
    test_ds = SatelliteDataset(split="test")
    
    # Limit visualization to first 10 examples
    num_samples = 10
    indices = range(num_samples)
    
    print(f"Generating 'Ground Truth vs Prediction' report for {num_samples} images...")
    
    for i in tqdm(indices):
        # Get Data from Dataset
        image_tensor, target = test_ds[i]
        
        # Prepare Image for OpenCV (C, H, W) -> (H, W, C) -> Un-normalize -> BGR
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # --- A. INFERENCE (PREDICTION) ---
        with torch.no_grad():
            # Model needs list of tensors
            preds = model([image_tensor.to(device)])
        prediction = preds[0]
        
        # --- B. VISUALIZE ---
        
        # 1. Original Image (Raw)
        img_raw = img_bgr.copy()
        cv2.putText(img_raw, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 2. Ground Truth (Expected)
        img_gt = visualize_instance_segmentation(
            img_bgr, 
            target['masks'], 
            target['boxes'], 
            scores=None, # No scores for GT
            title="Ground Truth"
        )
        
        # 3. Prediction (Model Output)
        img_pred = visualize_instance_segmentation(
            img_bgr, 
            prediction['masks'], 
            prediction['boxes'], 
            prediction['scores'], 
            threshold=0.5,
            title="Prediction"
        )
        
        # --- C. SAVE ---
        # Stack them horizontally: Original | Truth | Pred
        combined = np.hstack((img_raw, img_gt, img_pred))
        
        filename = f"comparison_{i}.jpg"
        cv2.imwrite(str(output_dir / filename), combined)

    print(f"\n✅ Done! Open {output_dir.absolute()} to see the comparisons.")

if __name__ == "__main__":
    main()