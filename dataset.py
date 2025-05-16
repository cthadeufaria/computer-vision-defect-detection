import json
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DTUDataset(Dataset):
    def __init__(self):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to the annotation file in COCO format (e.g., from GitHub).
            transform (callable, optional): Optional transform to be applied on an image.
        """
        super().__init__()
        annotations_dir = "DTU-annotations-main/"
        dataset_zip = "DTU - Drone inspection images of wind turbine/"

        if not os.path.exists(annotations_dir):
            os.system('wget https://github.com/imadgohar/DTU-annotations/archive/refs/heads/main.zip')
            os.system('unzip -o main.zip')
            os.system('rm main.zip')

        if not os.path.exists(dataset_zip):
            os.system('wget https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hd96prn3nc-2.zip')
            os.system('unzip -o hd96prn3nc-2.zip')
            os.system('rm hd96prn3nc-2.zip')

        os.system('mv "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/Nordtank 2017"/*.JPG "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"')
        os.system('mv "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/Nordtank 2018"/*.JPG "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"')

        self.root_dir = "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/"
        self.annotations = self.load_annotations("DTU-annotations-main/re-annotation/D3/train.json")

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=45),
            # transforms.RandomHorizontalFlip(p=0.5),
        ])

    def load_annotations(self, annotation_file):
        with open(annotation_file, 'r') as f:
            return json.load(f)

    def slice_image(self, image_path, tile_size=1024):
        try:
            row, col = map(int, image_path.split('/')[-1].split('.')[0].split('_')[-2:])
            new_path = image_path.split('.')[0][:-4] + '.JPG'
            image = Image.open(new_path).convert('RGB')
            width, height = image.size

            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size

            if right > width or lower > height:
                raise ValueError(f"Slice ({row}, {col}) exceeds image bounds: {width}x{height}")

            return image.crop((left, upper, right, lower))

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def __len__(self):
        return len(self.annotations['images'])

    def __getitem__(self, idx):
        img_info = self.annotations['images'][idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        slc = self.slice_image(img_path)

        if self.transform:
            image = self.transform(slc)

        ann_ids = [ann for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id']]
        boxes = [ann['bbox'] for ann in ann_ids]
        labels = [ann['category_id'] for ann in ann_ids]

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        return image, target
