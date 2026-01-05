"""
Wind Turbine Defect Detection Training Script
Fine-tunes a Faster R-CNN model on the DTU Wind Turbine dataset.

Optimized for Apple Silicon with MPS backend.
"""

import torch
import os
import glob
from torch.utils.data import DataLoader, random_split
from dataset import DTUDataset
from trainer import Trainer
from model import FasterRCNNModel
from torchmetrics.detection import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

torch.manual_seed(17)


def get_next_model_version(models_dir="./models"):
    """Find the next available version number for model saving."""
    pattern = os.path.join(models_dir, "faster_rcnn_wind_turbine_v*.pth")
    existing = glob.glob(pattern)

    if not existing:
        return 1

    # Extract version numbers and find max
    versions = []
    for path in existing:
        try:
            v = int(path.split("_v")[-1].split(".pth")[0])
            versions.append(v)
        except ValueError:
            continue

    return max(versions) + 1 if versions else 1


def collate_fn(batch):
    """Custom collate function for object detection."""
    return tuple(zip(*batch))


# Enable performance optimizations
torch.backends.cudnn.benchmark = True  # For CUDA
if hasattr(torch.backends, 'mps'):
    # MPS optimizations for Apple Silicon
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Prevent OOM

# Defect categories (shifted by +1 for Faster R-CNN)
CATEGORIES = {
    1: "VG;MT (Vortex Generator / Missing Tape)",
    2: "LE;ER (Leading Edge Erosion)",
    3: "LR;DA (Lightning Receptor Damage)",
    4: "LE;CR (Leading Edge Crack)",
    5: "SF;PO (Surface Pollution)"
}

def main():
    # NOTE: MPS (Apple Silicon GPU) hangs with Faster R-CNN due to unsupported ops
    # Using CPU for now. For GPU acceleration, use CUDA or switch to YOLO.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: Using CPU. Training will be slower but stable.")

    print(f'Using {device} for training')
    print(f"\nDefect Categories:")
    for k, v in CATEGORIES.items():
        print(f"  {k}: {v}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = DTUDataset()
    print(f"Dataset size: {len(dataset)} images")

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")

    # Create dataloaders - using single process to avoid pickle issues
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,  # Reduced for CPU memory
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Paths with versioning
    models_folder = "./models"
    runs_folder = "./runs"
    os.makedirs(models_folder, exist_ok=True)
    os.makedirs(runs_folder, exist_ok=True)

    version = get_next_model_version(models_folder)
    model_path = os.path.join(models_folder, f"faster_rcnn_wind_turbine_v{version}.pth")
    log_dir = os.path.join(runs_folder, f"v{version}")

    # Initialize model with 6 classes (5 defects + background)
    model = FasterRCNNModel(num_classes=6)
    trainer = Trainer(device, model, log_dir=log_dir)

    # Optimizer with lower learning rate for fine-tuning
    optimizer = Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)

    # Learning rate scheduler
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training
    num_epochs = 30  # Recommended: 30-50 epochs for good results

    print(f"\n{'='*50}")
    print(f"Starting training for {num_epochs} epochs...")
    print(f"{'='*50}\n")

    trainer.train_model(scheduler, optimizer, train_dataloader, num_epochs=num_epochs)

    # Save model with versioning
    torch.save(trainer.model.state_dict(), model_path)
    print(f"\nModel saved to {model_path} (version {version})")

    # Evaluation
    print(f"\n{'='*50}")
    print("Evaluating model on validation set...")
    print(f"{'='*50}\n")

    metric = MeanAveragePrecision().to(device)
    trainer.evaluate(metric, val_dataloader, num_epochs=1, log_epoch=num_epochs)

    # Close TensorBoard writer
    trainer.close()

    print("\nTraining complete!")
    print(f"View metrics with: tensorboard --logdir {runs_folder}")

if __name__ == "__main__":
    main()
