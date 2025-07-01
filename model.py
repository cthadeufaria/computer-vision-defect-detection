import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class FasterRCNNModel(nn.Module):
    """
    A wrapper class for the Faster R-CNN model with a ResNet backbone and FPN.
    This class initializes the model with a ResNet-50 backbone and an anchor generator.
    """ 
    def __init__(self):
        super(FasterRCNNModel, self).__init__()

        self.model = fasterrcnn_resnet50_fpn_v2(
            weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

    def forward(self, *x):
        return self.model(*x)