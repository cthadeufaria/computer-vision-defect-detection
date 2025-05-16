import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import DTUDataset
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from trainer import Trainer
from model import FasterRCNNModel


torch.backends.cudnn.benchmark = True
torch.manual_seed(17)
torch.cuda.manual_seed(17)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Using {device} for inference')

    dataset = DTUDataset()
    train_dataset, test_dataset = random_split(dataset, [int(len(dataset) * 0.8), int(len(dataset) * 0.2)])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    model = FasterRCNNModel()

    trainer = Trainer(device, model)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    model_path = "./mask_rcnn_dtu.pth"

    if not os.path.exists(model_path):
        trainer.train_model(optimizer, train_dataloader, num_epochs=20, device=device)

        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    else:
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        else:
            model.load_state_dict(torch.load(model_path))

        model.to(device)
        print(f"Model loaded from {model_path}")


    data_iter = iter(train_dataloader)
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
    plt.show()