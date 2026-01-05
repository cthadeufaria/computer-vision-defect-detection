import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os


class Trainer:
    def __init__(self, device, model, log_dir=None):
        """
        Initialize the Trainer with a model, optimizer, and dataloader.

        Args:
            model (nn.Module): The Mask R-CNN model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
            log_dir (str): Directory for TensorBoard logs. If None, creates timestamped dir.

        Returns:
            None
        """
        self.device = device
        self.model = model
        self.model.to(self.device)

        # Setup TensorBoard logging
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join("./runs", f"train_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        print(f"TensorBoard logs: {log_dir}")
        self.global_step = 0

    def train_model(self, scheduler, optimizer, dataloader, num_epochs):
        """
        Train the Mask R-CNN model with a specified dataloader and optimizer.

        Args:
            model (nn.Module): The Mask R-CNN model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            dataloader (torch.utils.data.DataLoader): Dataloader for the training data.
            num_epochs (int): Number of training epochs.
            device (str): Device to use for training (e.g., 'cuda' or 'cpu').

        Returns:
            None
        """
        self.model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            learning_rate = optimizer.param_groups[0]['lr']
            print(f'Current learning rate = {learning_rate:.6f}')

            # Only log CUDA memory if available
            if torch.cuda.is_available() and self.device.type == 'cuda':
                cuda_logs = torch.cuda.memory_stats(self.device)
                print(f"CUDA Memory Allocated (Peak): {cuda_logs['allocated_bytes.all.peak'] / 1e9:.4f} GB")
                print(f"CUDA Memory Reserved (Peak): {cuda_logs['reserved_bytes.all.peak'] / 1e9:.4f} GB")

            batch_count = 0
            for images, targets in dataloader:
                targets = self.xyhw2xyxy(targets)

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                try:
                    loss_dict = self.model(images, targets)

                    loss = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                    # Log batch losses to TensorBoard
                    self.writer.add_scalar('Loss/batch_total', loss.item(), self.global_step)
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'Loss/batch_{k}', v.item(), self.global_step)
                    self.global_step += 1

                    loss_logs = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                    print(f"Batch Loss: {loss.item():.4f} | {loss_logs}")

                except AssertionError as e:
                    print(f"Skipping batch due to error: {e}")
                    continue

            scheduler.step()

            # Log epoch metrics to TensorBoard
            avg_epoch_loss = epoch_loss / max(batch_count, 1)
            self.writer.add_scalar('Loss/epoch_total', epoch_loss, epoch + 1)
            self.writer.add_scalar('Loss/epoch_average', avg_epoch_loss, epoch + 1)
            self.writer.add_scalar('Learning_Rate', learning_rate, epoch + 1)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {epoch_loss:.4f}")

    def evaluate(self, metric, dataloader, num_epochs, log_epoch=None):
        self.model.eval()

        for epoch in range(num_epochs):
            with torch.no_grad():
                for images, targets in dataloader:
                    targets = self.xyhw2xyxy(targets)

                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    predictions = self.model(images)

                    metric.update(predictions, targets)

            results = metric.compute()

            # Log evaluation metrics to TensorBoard
            eval_epoch = log_epoch if log_epoch is not None else epoch + 1
            self.writer.add_scalar('Metrics/mAP', results['map'].item(), eval_epoch)
            self.writer.add_scalar('Metrics/mAP_50', results['map_50'].item(), eval_epoch)
            self.writer.add_scalar('Metrics/mAP_75', results['map_75'].item(), eval_epoch)

            print(f"\nEpoch {epoch + 1}/{num_epochs} - Evaluation Results:")
            print(f"\nmAP: {results['map']:.4f} ----------------------------\n")
            print(f"mAP@0.5: {results['map_50']:.4f} ----------------------------\n")

        return results

    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()

    def xyhw2xyxy(self, targets):
        """Convert bounding boxes from [x, y, width, height] to [x_min, y_min, x_max, y_max]"""
        for target in targets:
            boxes = target['boxes'].numpy()
            if len(boxes) == 0:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                continue
            converted_boxes = []
            for box in boxes:
                x_min, y_min, width, height = box
                x_max = x_min + width
                y_max = y_min + height
                converted_boxes.append([x_min, y_min, x_max, y_max])
            target['boxes'] = torch.tensor(converted_boxes, dtype=torch.float32)

        return targets