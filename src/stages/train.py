import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Import both architecture options
from src.models.custom_architecture import get_custom_model
from src.models.transfer_learning import get_transfer_model
from src.utils.dataset import SatelliteDataset

# ==========================================================================================
# HELPER FUNCTIONS FOR METRICS
# ==========================================================================================

def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Executes one training epoch.
    Returns: Average loss for the epoch.
    """
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        # Move data to GPU
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward Pass (Calculates loss automatically in train mode)
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward Pass
        optimizer.zero_grad()
        losses.backward()
        
        # Gradient Clipping (Prevents exploding gradients in custom layers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # Logging
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        wandb.log({"train_batch_loss": losses.item()})

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, metric_calc):
    """
    Evaluates the model on the validation set.
    Returns: map_stats (mAP dictionary from torchmetrics)
    """
    model.eval()
    metric_calc.reset()
    
    pbar = tqdm(loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Inference
        outputs = model(images)
        
        # Format predictions for TorchMetrics (needs uint8 binary masks)
        formatted_preds = []
        for pred in outputs:
            # Threshold soft probabilities to binary mask (0 or 1)
            masks = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)
            formatted_preds.append({
                "boxes": pred['boxes'], 
                "scores": pred['scores'], 
                "labels": pred['labels'], 
                "masks": masks
            })
            
        # Format targets
        formatted_targets = []
        for t in targets:
            formatted_targets.append({
                "boxes": t['boxes'], 
                "labels": t['labels'], 
                "masks": t['masks'].to(torch.uint8)
            })

        # Update metric calculator
        metric_calc.update(formatted_preds, formatted_targets)

    # Compute final mAP scores across the whole dataset
    map_results = metric_calc.compute()
    return map_results

# ==========================================================================================
# MAIN TRAINING PIPELINE
# ==========================================================================================

def main():
    # 1. Load Configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Initialize Weights & Biases
    wandb.init(
        project=params["base"]["project"],
        config=params,
        name=f"{params['train']['model_type']}_run"
    )

    # 3. Prepare Datasets
    # The SatelliteDataset class handles loading images and applying augmentations
    print("Loading Datasets...")
    train_ds = SatelliteDataset(split="train")
    val_ds = SatelliteDataset(split="val")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=params["train"]["batch_size"], 
        shuffle=True, 
        num_workers=params["train"]["workers"], 
        collate_fn=SatelliteDataset.collate_fn
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=params["train"]["batch_size"], 
        shuffle=False, 
        num_workers=params["train"]["workers"], 
        collate_fn=SatelliteDataset.collate_fn
    )

    # 4. Initialize Model
    # Switches between Transfer Learning (1pk) and Custom Architecture (2pk)
    model_type = params["train"]["model_type"]
    print(f"Initializing Model: {model_type.upper()}")
    
    if model_type == "transfer":
        model = get_transfer_model(num_classes=2).to(device)
        cfg = params["train"]["transfer_config"]
    else:
        # Custom Architecture (>50% own layers)
        model = get_custom_model(num_classes=2).to(device)
        cfg = params["train"]["custom_config"]

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=cfg["learning_rate"]
    )
    
    # Reduce Learning Rate if validation mAP stagnates
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2
    )
    
    # Metric Calculator (IoU Type = 'segm' for masks)
    metric_calc = MeanAveragePrecision(iou_type="segm").to(device)
    
    # Weights saving directory
    save_dir = Path(f"runs/{model_type}_model/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_map = 0.0

    # 6. Training Loop
    print("Starting Training Loop...")
    for epoch in range(params["train"]["epochs"]):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        
        # Evaluate
        map_stats = evaluate(model, val_loader, device, metric_calc)
        map_50 = map_stats['map_50'].item() # mAP at IoU=0.50
        
        # Get Current Learning Rate
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Step Scheduler
        scheduler.step(map_50)
        
        # Free up GPU memory
        torch.cuda.empty_cache()
        
        # Console Log
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | mAP@50: {map_50:.4f} | LR: {current_lr:.6f}")
        
        # WandB Log (Includes Learning Rate!)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mAP_50": map_50,
            "val_mAP_COCO": map_stats['map'].item(),
            "learning_rate": current_lr 
        })
        
        # Save Best Model
        if map_50 > best_map:
            best_map = map_50
            torch.save(model.state_dict(), save_dir / "best.pth")
            print(f"Saved new best model (mAP@50: {best_map:.4f})")
            
    wandb.finish()

if __name__ == "__main__":
    main()