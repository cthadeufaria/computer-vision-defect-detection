import torch


class Trainer:
    def __init__(self, device, model):
        """
        Initialize the Trainer with a model, optimizer, and dataloader.

        Args:
            model (nn.Module): The Mask R-CNN model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            dataloader (torch.utils.data.DataLoader): Dataloader for the training data.

        Returns:
            None
        """
        self.device = device
        self.model = model
        self.model.to(self.device)

    def train_model(self, optimizer, dataloader, num_epochs=20):
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
            for images, targets in dataloader:
                for target in targets:
                    array = target['boxes'].numpy()
                    x_min, y_min, width, height = array[0]
                    x_max = x_min + width
                    y_max = y_min + height
                    target['boxes'] = torch.tensor([[
                        x_min, y_min, x_max, y_max
                    ]], dtype=torch.float32)

                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                try:
                    loss_dict = self.model(images, targets)

                    loss = sum(loss for loss in loss_dict.values())

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    loss_logs = " | ".join([f"{k}: {v.item():.4f}" for k, v in loss_dict.items()])
                    print(f"Batch Loss: {loss.item():.4f} | {loss_logs}")

                except AssertionError as e:
                    print(f"Skipping batch due to error: {e}")
                    continue

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Loss: {epoch_loss:.4f}")

    def evaluation(self, dataloader):
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                predictions = self.model(images)

                for i, prediction in enumerate(predictions):
                    print(f"Image {i}:")
                    print(f"  Boxes: {prediction['boxes']}")
                    print(f"  Labels: {prediction['labels']}")
                    print(f"  Scores: {prediction['scores']}")

                    # Calculate loss
                    loss = criterion(prediction['scores'], targets[i]['labels'])
                    total_loss += loss.item()

                    # Calculate accuracy
                    _, predicted_labels = torch.max(prediction['scores'], 1)
                    total_correct += (predicted_labels == targets[i]['labels']).sum().item()
                    total_samples += targets[i]['labels'].size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples * 100

        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}%")