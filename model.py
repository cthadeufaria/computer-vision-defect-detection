import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class FasterRCNNModel(nn.Module):
    """
    A wrapper class for the Faster R-CNN model with a ResNet backbone and FPN.
    This class initializes the model with a ResNet-50 backbone and an anchor generator.
    Fine-tuned for wind turbine defect detection with 5 defect categories + background.
    """
    def __init__(self, num_classes=6):
        super(FasterRCNNModel, self).__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

        # Replace the classifier head for our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, *x):
        return self.model(*x)