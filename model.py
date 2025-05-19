import torch
import torch.nn as nn
# from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2


class FasterRCNNModel(nn.Module):
    """
    A wrapper class for the Faster R-CNN model with a ResNet backbone and FPN.
    This class initializes the model with a ResNet-50 backbone and an anchor generator.
    """ 
    def __init__(self):
        super(FasterRCNNModel, self).__init__()

        # anchor_generator = AnchorGenerator(
        # sizes=((32,), (64,), (128,), (256,), (512,)),
        # aspect_ratios=((0.5, 1.0, 2.0),) * 5)

        # num_classes = 6

        # self.model = fasterrcnn_resnet50_fpn(
        #     weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        # )

        self.model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        )

    def forward(self, *x):
        image_size = x[0][0].shape[-2:]
        bboxes = [d['boxes'] for d in x[1]]
        masks = self.create_mask_from_boxes(image_size, bboxes)
        
        for i, value in enumerate(x[1]):
            value['masks'] = masks[i]

        return self.model(*x)
    
    def create_mask_from_boxes(self, image_size, boxes):
        masks = []

        for box in boxes:
            mask = torch.zeros(image_size, dtype=torch.uint8)
            x1, y1, x2, y2 = box.int().tolist()[0]
            mask[y1:y2, x1:x2] = 1
            masks.append(mask)

        return torch.stack(masks)