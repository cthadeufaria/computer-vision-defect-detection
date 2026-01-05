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
            annotation_file (str): Path to the annotation file in COCO format.
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

        # Images are in Nordtank 2018 subdirectory
        self.root_dir = "DTU - Drone inspection images of wind turbine/DTU - Drone inspection images of wind turbine/Nordtank 2018/"
        self.annotations = self.load_annotations("DTU-annotations-main/re-annotation/D3/train.json")

        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Validate images and filter out missing ones
        self.valid_indices, self.missing_images = self._validate_images()
        print(f"Valid images: {len(self.valid_indices)}, Missing images: {len(self.missing_images)}")

        # Save missing images to file
        if self.missing_images:
            with open("missing_images.txt", "w") as f:
                for img_name in self.missing_images:
                    f.write(f"{img_name}\n")
            print(f"Missing images saved to missing_images.txt")

    def _validate_images(self):
        """Check which images exist and return valid indices and missing image names."""
        valid_indices = []
        missing_images = []

        for idx, img_info in enumerate(self.annotations['images']):
            # Extract base image name from sliced filename (e.g., DJI_0168_1_2.JPG -> DJI_0168.JPG)
            file_name = img_info['file_name']
            base_name = file_name.split('.')[0][:-4] + '.JPG'  # Remove _X_X suffix
            base_path = os.path.join(self.root_dir, base_name)

            if os.path.exists(base_path):
                valid_indices.append(idx)
            else:
                missing_images.append(f"{file_name} (base: {base_name})")

        return valid_indices, missing_images

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
        # Return only valid images count
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Map to valid index
        actual_idx = self.valid_indices[idx]
        img_info = self.annotations['images'][actual_idx]
        img_path = os.path.join(self.root_dir, img_info['file_name'])
        slc = self.slice_image(img_path)

        if slc is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        if self.transform:
            image = self.transform(slc)

        ann_ids = [ann for ann in self.annotations['annotations'] if ann['image_id'] == img_info['id']]
        boxes = [ann['bbox'] for ann in ann_ids]
        # Shift labels by 1 since Faster R-CNN reserves 0 for background
        labels = [ann['category_id'] + 1 for ann in ann_ids]

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        return image, target
