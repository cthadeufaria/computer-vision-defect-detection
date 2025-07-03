import torch
import os
import numpy as np
from torch.utils.data import DataLoader, random_split
from dataset import DTUDataset
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from trainer import Trainer
from model import FasterRCNNModel
from torchmetrics.detection import MeanAveragePrecision
from datetime import datetime
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam

torch.backends.cudnn.benchmark = True
torch.manual_seed(17)
torch.cuda.manual_seed(17)


def plot_next_image_with_bboxes(dataloader):
    data_iter = iter(dataloader)
    images, targets = next(data_iter)

    image = images[0]
    target = targets[0]
    array = image.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(array)

    for box in target['boxes']:
        x_min, y_min, width, height = box
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.imshow(array)
    plt.title(f"Labels: {target['labels']}, Bboxes: {target['boxes']}")

    now = datetime.now()
    plt.savefig(f"./figures/{now}_label_bbox.jpg")

def plot_next_image_with_predictions(device, dataloader, model):
    val_iter = iter(dataloader)

    model.eval()

    for _ in range(10):
        val_image, val_target = next(val_iter)

        val_image = [img.to(device) for img in val_image]

        with torch.no_grad():
            prediction = model(val_image)

        val_image_np = val_image[0].permute(1, 2, 0).cpu().numpy()
        val_image_np = val_image_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
        val_image_np = np.clip(val_image_np, 0, 1)

        fig, ax = plt.subplots(1)
        ax.imshow(val_image_np)

        for box, label in zip(prediction[0]['boxes'], prediction[0]['labels']):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, str(label.item()), color='red', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))

        for box, label in zip(val_target[0]['boxes'], val_target[0]['labels']):
            x_min, y_min, width, height = box.cpu().numpy()
            rect = patches.Rectangle((x_min, y_min), width, height, 
                                linewidth=1, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min, str(label.item()), color='blue', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.7))

        plt.title("Predicted (Red) vs. Annotated (Blue) Bounding Boxes")
        plt.axis('off')

        now = datetime.now()
        plt.savefig(f"./figures/{now}_pred_label_bbox.jpg")

def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    dataset = DTUDataset()

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = FasterRCNNModel()

    trainer = Trainer(device, model)

    optimizer = Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)

    scheduler = ExponentialLR(optimizer, gamma=0.95)

    folder = "./models"
    model_path = os.path.join(folder, "faster_rcnn.pth")

    if not os.path.exists(model_path):
        trainer.train_model(scheduler, optimizer, train_dataloader, num_epochs=500)

        torch.save(trainer.model.state_dict(), model_path)

        x = next(iter(train_dataloader))[0]
        x = [img.to(device) for img in x]
        
        torch.onnx.export(
            model,
            x,
            os.path.join(folder, "faster_rcnn.onnx"),
            input_names=["input"],
            output_names=["boxes", "labels", "scores"]
        )

        print(f"Model saved to {model_path}")

    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

        print(f"Model loaded from {model_path}")

    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    metric = MeanAveragePrecision().to(device)

    plot_next_image_with_predictions(device, val_dataloader, model)

    trainer.evaluate(metric, val_dataloader, num_epochs=50)


if __name__ == "__main__":
    main()