import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence

class SoccerDatasetSequence(Dataset):
    """
    A Pytorch Dataset class adapted for SoccerNet tracking data, loading all data upfront.

    Returns (from __getitem__):
    ------------
    image: torch.Tensor of size (C, H, W)
    gt_bboxes: torch.Tensor of size (max_objects, 4) [ymin, xmin, ymax, xmax] padded with -1
    gt_classes: torch.Tensor of size (max_objects,) [class_id] padded with -1 (all valid objects are class 1)
    """
    def __init__(self, img_size, sequence_dir):
        self.sequence_dir = sequence_dir
        self.img_size = img_size
        self.img_dir = os.path.join(sequence_dir, "img1")
        self.gt_txt = os.path.join(sequence_dir, "gt", "gt.txt")

       
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

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
            # Use [ymin, xmin, ymax, xmax] format
            self.annotations[frame_id]["boxes"].append([ xmin, ymin, xmax, ymax ])
            # We don't use track_id here, will set class to 1 later


        self.img_data_all, self.gt_bboxes_all, self.gt_classes_all = self._load_all_data()

    def __len__(self):
        # Return the number of loaded images
        return self.img_data_all.size(0)

    def __getitem__(self, idx):
        # Return the pre-loaded, pre-processed data for the given index
        return self.img_data_all[idx], self.gt_bboxes_all[idx], self.gt_classes_all[idx]

    def _load_all_data(self):
        img_data_list = []
        gt_boxes_list = []
        gt_classes_list = []

        for img_name in self.img_files:
            frame_idx = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(self.img_dir, img_name)

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"Warning: Error loading/transforming image {img_path}: {e}, skipping.")
                continue

            # Load annotations for this frame
            ann = self.annotations.get(frame_idx, {"boxes": [], "labels": []})
            boxes = torch.tensor(ann["boxes"], dtype=torch.float32)

            # Create class labels: 1 for every valid box
            if boxes.nelement() > 0: # Check if there are any boxes
                 labels = torch.ones(boxes.shape[0], dtype=torch.int64)
            else:
                 labels = torch.empty((0,), dtype=torch.int64) # Empty tensor if no boxes


            img_data_list.append(image_tensor)
            gt_boxes_list.append(boxes)
            gt_classes_list.append(labels)

        if not img_data_list:
             raise ValueError("No valid images found or processed.")


        # Stack images
        img_data_stacked = torch.stack(img_data_list, dim=0)

        # Pad bounding boxes and classes sequences
        # Note: padding_value=-1 matches the original ObjectDetectionDataset example
        gt_bboxes_pad = pad_sequence(gt_boxes_list, batch_first=True, padding_value=-1)
        gt_classes_pad = pad_sequence(gt_classes_list, batch_first=True, padding_value=-1)

        return img_data_stacked.to(dtype=torch.float32), gt_bboxes_pad, gt_classes_pad


# Removed the custom collate_fn as the dataset now returns stacked/padded tensors


def get_dataloader(
    sequence_dir,
    img_size, # Added img_size argument
    batch_size=2,
    shuffle=False,
    num_workers=0
    # Removed transforms argument
):
    """
    Create a PyTorch DataLoader for soccer tracking sequences.

    Args:
        sequence_dir (str): Path to a SNMOT-XXX sequence directory.
        img_size (int or tuple): The target size (H, W) for the images.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data each epoch.
        num_workers (int): Number of worker processes for loading.

    Returns:
        DataLoader: an iterable over (images, gt_bboxes, gt_classes) tuples.
                  images:     Tensor [B, C, H, W]
                  gt_bboxes:  Tensor [B, max_objs, 4]
                  gt_classes: Tensor [B, max_objs]
    """
    # Pass img_size to the dataset constructor
    dataset = SoccerDatasetSequence(img_size=img_size, sequence_dir=sequence_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # Use default collate_fn since dataset handles stacking/padding
        collate_fn=None
    )
    return loader 