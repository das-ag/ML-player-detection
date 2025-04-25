import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence

class SoccerDatasetSequence(Dataset):
    """
    A PyTorch Dataset class for a single sequence from the SoccerNet dataset.
    
    Returns from __getitem__:
        image:     Tensor [C, H, W]
        gt_bboxes: Tensor [num_objects, 4] as [xmin, ymin, xmax, ymax]
        gt_classes:Tensor [num_objects] (all ones)
    """
    def __init__(self, img_size, sequence_dir):
        self.img_size = img_size
        self.img_dir = os.path.join(sequence_dir, "img1")
        gt_txt = os.path.join(sequence_dir, "gt", "gt.txt")

        
        gt_data = np.loadtxt(gt_txt, delimiter=',')
        ann_dict = {}
        for row in gt_data:
            frame_id = int(row[0])
            x, y, w, h = row[2], row[3], row[4], row[5]
            xmin, ymin = float(x), float(y)
            xmax, ymax = float(x + w), float(y + h)
            ann_dict.setdefault(frame_id, []).append([xmin, ymin, xmax, ymax])

        # Build list of (img_path, boxes_tensor, labels_tensor)
        img_names = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png"))
        )
        items = []
        max_objs = 0
        
        
        for name in img_names:
            frame_idx = int(os.path.splitext(name)[0])
            boxes = ann_dict.get(frame_idx, [])
            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.ones(boxes_tensor.size(0), dtype=torch.int64)
            else:
                boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
                labels_tensor = torch.empty((0,), dtype=torch.int64)
            items.append((os.path.join(self.img_dir, name), boxes_tensor, labels_tensor))
            max_objs = max(max_objs, boxes_tensor.size(0))

        self.items = items
        self.max_objs = max_objs
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, boxes, labels = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, boxes, labels

def get_dataloader(
    sequence_dir,
    img_size,
    batch_size=2,
    shuffle=False,
    num_workers=0
):
    """
    Create a DataLoader yielding:
      images:     Tensor [B, C, H, W]
      gt_bboxes:  Tensor [B, max_objs, 4]
      gt_classes: Tensor [B, max_objs]
    """
    dataset = SoccerDatasetSequence(img_size=img_size, sequence_dir=sequence_dir)

    def collate_fn(batch):
        imgs, bboxes, classes = zip(*batch)
        images = torch.stack(imgs, dim=0)
        bboxes_padded = pad_sequence(bboxes, batch_first=True, padding_value=-1)
        classes_padded = pad_sequence(classes, batch_first=True, padding_value=-1)
        # Ensure consistent padding to the dataset‐wide max_objs
        if bboxes_padded.size(1) < dataset.max_objs:
            extra = dataset.max_objs - bboxes_padded.size(1)
            bboxes_padded = F.pad(bboxes_padded, (0, 0, 0, extra), value=-1)
            classes_padded = F.pad(classes_padded, (0, extra), value=-1)
        return images, bboxes_padded, classes_padded

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader