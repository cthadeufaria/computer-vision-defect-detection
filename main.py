"""
Wind Turbine Defect Detection - Inference Script

Run inference on images using a trained Faster R-CNN model.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --input_dir path/to/images/ --output_dir path/to/results/
    python main.py --generate_samples  # Generate sample detections for README
"""

import torch
import os
import argparse
import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
from model import FasterRCNNModel
from datetime import datetime

# Defect category names
CATEGORIES = {
    0: "Background",
    1: "VG;MT (Vortex Generator/Missing Tape)",
    2: "LE;ER (Leading Edge Erosion)",
    3: "LR;DA (Lightning Receptor Damage)",
    4: "LE;CR (Leading Edge Crack)",
    5: "SF;PO (Surface Pollution)"
}

# Colors for each category (RGB)
COLORS = {
    1: (255, 0, 0),      # Red
    2: (0, 255, 0),      # Green
    3: (0, 0, 255),      # Blue
    4: (255, 165, 0),    # Orange
    5: (128, 0, 128),    # Purple
}


def get_latest_model(models_dir="./models"):
    """Find the latest versioned model file."""
    pattern = os.path.join(models_dir, "faster_rcnn_wind_turbine_v*.pth")
    model_files = glob.glob(pattern)

    if not model_files:
        # Try non-versioned filename
        fallback = os.path.join(models_dir, "faster_rcnn_wind_turbine.pth")
        if os.path.exists(fallback):
            return fallback
        raise FileNotFoundError(f"No model found in {models_dir}")

    # Sort by version number
    model_files.sort(key=lambda x: int(x.split("_v")[-1].split(".pth")[0]))
    return model_files[-1]


def load_model(model_path=None, device="cpu"):
    """Load the trained model."""
    if model_path is None:
        model_path = get_latest_model()

    print(f"Loading model from: {model_path}")

    model = FasterRCNNModel(num_classes=6)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def preprocess_image(image_path):
    """Load and preprocess an image for inference."""
    image = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.Resize((1024, 1024)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tensor = transform(image)
    return tensor, image


def run_inference(model, image_tensor, device, confidence_threshold=0.5):
    """Run inference on a single image."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]

    # Filter by confidence threshold
    mask = pred['scores'] >= confidence_threshold

    return {
        'boxes': pred['boxes'][mask].cpu().numpy(),
        'labels': pred['labels'][mask].cpu().numpy(),
        'scores': pred['scores'][mask].cpu().numpy()
    }


def visualize_predictions(image, predictions, output_path=None, show=False):
    """Visualize predictions on an image."""
    fig, ax = plt.subplots(1, figsize=(12, 12))

    # Convert PIL image to numpy for display
    img_array = np.array(image.resize((1024, 1024)))
    ax.imshow(img_array)

    # Draw predictions
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # Get color for this category
        color = np.array(COLORS.get(label, (255, 255, 255))) / 255.0

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Add label with score
        label_text = f"{CATEGORIES.get(label, 'Unknown')}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8)
        )

    ax.axis('off')
    ax.set_title(f"Detected {len(predictions['boxes'])} defect(s)", fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")

    if show:
        plt.show()

    plt.close()


def process_single_image(model, image_path, output_path, device, confidence_threshold=0.5):
    """Process a single image and save results."""
    print(f"\nProcessing: {image_path}")

    # Load and preprocess
    image_tensor, original_image = preprocess_image(image_path)

    # Run inference
    predictions = run_inference(model, image_tensor, device, confidence_threshold)

    # Print results
    print(f"  Found {len(predictions['boxes'])} defect(s):")
    for label, score in zip(predictions['labels'], predictions['scores']):
        print(f"    - {CATEGORIES.get(label, 'Unknown')}: {score:.2%}")

    # Visualize
    visualize_predictions(original_image, predictions, output_path)

    return predictions


def process_directory(model, input_dir, output_dir, device, confidence_threshold=0.5):
    """Process all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Find all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    all_results = {}
    for image_path in image_files:
        basename = os.path.basename(image_path)
        name, _ = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_detection.png")

        predictions = process_single_image(model, image_path, output_path, device, confidence_threshold)
        all_results[basename] = predictions

    print(f"\nProcessed {len(image_files)} images. Results saved to {output_dir}")
    return all_results


def generate_sample_detections(model, device, num_samples=3):
    """Generate sample detection images for README."""
    from dataset import DTUDataset
    from torch.utils.data import DataLoader

    print("Generating sample detection images...")

    # Load dataset
    dataset = DTUDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    os.makedirs("./figures", exist_ok=True)

    samples_generated = 0
    for i, (images, targets) in enumerate(loader):
        if samples_generated >= num_samples:
            break

        # Skip images with no annotations
        if len(targets[0]['boxes']) == 0:
            continue

        image_tensor = images[0]

        # Run inference
        predictions = run_inference(model, image_tensor, device, confidence_threshold=0.3)

        # Only save if we have predictions
        if len(predictions['boxes']) > 0:
            # Denormalize image for visualization
            img_np = image_tensor.permute(1, 2, 0).numpy()
            img_np = img_np * 0.5 + 0.5  # Denormalize
            img_np = np.clip(img_np, 0, 1)

            # Create figure with predictions and ground truth
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))

            # Left: Ground truth
            axes[0].imshow(img_np)
            axes[0].set_title("Ground Truth", fontsize=14)
            for box, label in zip(targets[0]['boxes'].numpy(), targets[0]['labels'].numpy()):
                x, y, w, h = box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='blue', facecolor='none')
                axes[0].add_patch(rect)
                axes[0].text(x, y - 5, CATEGORIES.get(label, 'Unknown'), color='white', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='blue', alpha=0.8))
            axes[0].axis('off')

            # Right: Predictions
            axes[1].imshow(img_np)
            axes[1].set_title("Model Predictions", fontsize=14)
            for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                x1, y1, x2, y2 = box
                color = np.array(COLORS.get(label, (255, 0, 0))) / 255.0
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
                axes[1].add_patch(rect)
                axes[1].text(x1, y1 - 5, f"{CATEGORIES.get(label, 'Unknown')}: {score:.0%}",
                           color='white', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
            axes[1].axis('off')

            plt.tight_layout()
            output_path = f"./figures/sample_detection_{samples_generated + 1}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {output_path}")
            samples_generated += 1

    print(f"\nGenerated {samples_generated} sample images in ./figures/")


def main():
    parser = argparse.ArgumentParser(description="Wind Turbine Defect Detection Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--input_dir", type=str, help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--model", type=str, default=None, help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (0-1)")
    parser.add_argument("--generate_samples", action="store_true", help="Generate sample detections for README")

    args = parser.parse_args()

    # Select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model
    try:
        model = load_model(args.model, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first using: python train_wind_turbine.py")
        return

    # Process based on arguments
    if args.generate_samples:
        generate_sample_detections(model, device)
    elif args.image:
        os.makedirs(args.output_dir, exist_ok=True)
        basename = os.path.basename(args.image)
        name, _ = os.path.splitext(basename)
        output_path = os.path.join(args.output_dir, f"{name}_detection.png")
        process_single_image(model, args.image, output_path, device, args.confidence)
    elif args.input_dir:
        process_directory(model, args.input_dir, args.output_dir, device, args.confidence)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python main.py --image test.jpg")
        print("  python main.py --input_dir ./images --output_dir ./results")
        print("  python main.py --generate_samples")


if __name__ == "__main__":
    main()
