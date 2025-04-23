import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

class SoccerDatasetSequence(Dataset):
    def __init__(self, sequence_dir, transforms=None):
        # Initialize dataset with sequence directory and transforms
        self.sequence_dir = sequence_dir
        self.transforms = transforms
        self.img_dir = os.path.join(sequence_dir, "img1")
        self.gt_txt = os.path.join(sequence_dir, "gt", "gt.txt")

        # List and sort image files
        self.img_files = sorted(
            [f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png"))]
        )

        # Load ground truth and organize per frame
        gt_data = np.loadtxt(self.gt_txt, delimiter=',')
        self.annotations = {}
        for row in gt_data:
            frame_id, track_id, x, y, w, h, *rest = row
            frame_id = int(frame_id)
            ymin = float(y)
            xmin = float(x)
            ymax = float(y + h)
            xmax = float(x + w)
            if frame_id not in self.annotations:
                self.annotations[frame_id] = {"boxes": [], "labels": []}
            # Use [ymin, xmin, ymax, xmax] format to match RPN expectations
            self.annotations[frame_id]["boxes"].append([ymin, xmin, ymax, xmax])
            self.annotations[frame_id]["labels"].append(int(track_id))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.img_files[idx]
        frame_idx = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)

        # Load annotations for this frame
        ann = self.annotations.get(frame_idx, {"boxes": [], "labels": []})
        boxes = torch.tensor(ann["boxes"], dtype=torch.float32)
        labels = torch.tensor(ann["labels"], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        return image, target


def collate_fn(batch):
    # Collate images and targets into lists for batching
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataloader(
    sequence_dir,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    transforms=None
):
    """
    Create a PyTorch DataLoader for soccer tracking sequences.

    Args:
        sequence_dir (str): Path to a SNMOT-XXX sequence directory.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data each epoch.
        num_workers (int): Number of worker processes for loading.
        transforms (callable, optional): Transformations applied to each image.

    Returns:
        DataLoader: an iterable over (images, targets) tuples.
    """
    dataset = SoccerDatasetSequence(sequence_dir, transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader 