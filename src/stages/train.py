import torch
import yaml
import wandb
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# --- IMPORT BOTH MODELS ---
from src.models.custom_architecture import get_custom_model
from src.models.transfer_learning import get_transfer_model  # <--- NEW IMPORT
from src.utils.dataset import SatelliteDataset

def calculate_iou(pred_masks, target_masks, threshold=0.5):
    """Calculates Pixel-wise IoU."""
    if len(pred_masks) == 0 and len(target_masks) == 0: return 1.0
    if len(pred_masks) == 0 or len(target_masks) == 0: return 0.0

    preds = (pred_masks > threshold).float().view(-1)
    targets = target_masks.float().view(-1)
    
    intersection = (preds * targets).sum()
    union = (preds + targets).sum() - intersection
    return 1.0 if union == 0 else (intersection / union).item()

def calculate_pixel_f1(pred_masks, target_masks, threshold=0.5):
    """Calculates Pixel-wise F1 Score."""
    if len(pred_masks) == 0 and len(target_masks) == 0: return 1.0
    if len(pred_masks) == 0 or len(target_masks) == 0: return 0.0

    preds = (pred_masks > threshold).float().view(-1)
    targets = target_masks.float().view(-1)
    
    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    return f1.item()

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += losses.item()
        pbar.set_postfix(loss=losses.item())
        wandb.log({"train_batch_loss": losses.item()})

    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, metric_calc):
    model.eval()
    metric_calc.reset()
    total_f1 = 0
    total_iou = 0
    batches = 0
    
    pbar = tqdm(loader, desc="Evaluating")
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        # Formatted preds for mAP
        formatted_preds = []
        for pred in outputs:
            masks = pred['masks']
            masks = (masks > 0.5).squeeze(1).to(torch.uint8)
            formatted_preds.append({
                "boxes": pred['boxes'],
                "scores": pred['scores'],
                "labels": pred['labels'],
                "masks": masks
            })
            
        formatted_targets = []
        for t in targets:
            formatted_targets.append({
                "boxes": t['boxes'],
                "labels": t['labels'],
                "masks": t['masks'].to(torch.uint8)
            })

        metric_calc.update(formatted_preds, formatted_targets)

        # Pixel Metrics
        for i, output in enumerate(outputs):
            pred_masks = output['masks']
            gt_masks = targets[i]['masks']
            
            if len(pred_masks) > 0: pred_map = torch.max(pred_masks, dim=0)[0]
            else: pred_map = torch.zeros((500, 500)).to(device)
                
            if len(gt_masks) > 0: gt_map = torch.max(gt_masks, dim=0)[0]
            else: gt_map = torch.zeros((500, 500)).to(device)

            total_iou += calculate_iou(pred_map, gt_map)
            total_f1 += calculate_pixel_f1(pred_map, gt_map)
            batches += 1

    map_results = metric_calc.compute()
    return total_iou / max(batches, 1), total_f1 / max(batches, 1), map_results

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    # --- SELECT CONFIG BASED ON MODEL TYPE ---
    model_type = params["train"].get("model_type", "custom")
    if model_type == "yolo": # Fallback if you used YOLO config names
        model_type = "transfer"
        
    if model_type == "transfer":
        # Use YOLO/Transfer config settings if available, else custom
        cfg = params["train"].get("yolo_config", params["train"]["custom_config"])
        run_name = "transfer_resnet50_run"
    else:
        cfg = params["train"]["custom_config"]
        run_name = "custom_architecture_run"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    wandb.init(
        project=params["base"]["project"],
        config=params,
        name=run_name
    )

    train_ds = SatelliteDataset(split="train")
    val_ds = SatelliteDataset(split="val")
    
    train_loader = DataLoader(
        train_ds, batch_size=params["train"]["batch_size"], shuffle=True,
        num_workers=params["train"]["workers"], collate_fn=SatelliteDataset.collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=params["train"]["batch_size"], shuffle=False, 
        num_workers=params["train"]["workers"], collate_fn=SatelliteDataset.collate_fn
    )

    # --- MODEL SELECTION LOGIC ---
    print(f"Initializing Model Type: {model_type.upper()}...")
    if model_type == "transfer":
        model = get_transfer_model(num_classes=2).to(device)
    else:
        model = get_custom_model(num_classes=2).to(device)
    
    # Optimizer (Use LR from config)
    lr = cfg.get("lr0", cfg.get("learning_rate", 0.001))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=lr
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=2
    )

    metric_calc = MeanAveragePrecision(iou_type="segm").to(device)
    save_dir = Path(f"runs/{model_type}_model/weights")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_map = 0.0
    print("Starting Training...")
    
    for epoch in range(params["train"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_iou, val_f1, val_map_stats = evaluate(model, val_loader, device, metric_calc)
        
        map_50 = val_map_stats['map_50'].item()
        map_coco = val_map_stats['map'].item()
        
        scheduler.step(map_50)
        curr_lr = optimizer.param_groups[0]["lr"]
        torch.cuda.empty_cache()
        
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | IoU: {val_iou:.4f} | mAP@50: {map_50:.4f}")
        
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_iou": val_iou,       
            "val_f1": val_f1,
            "val_mAP_50": map_50,
            "val_mAP_COCO": map_coco,
            "learning_rate": curr_lr
        })
        
        if map_50 > best_map:
            best_map = map_50
            torch.save(model.state_dict(), save_dir / "best.pth")
            
    wandb.finish()

if __name__ == "__main__":
    main()