import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

class SoccerDatasetSequence(Dataset):
    """
    A lazy-loading PyTorch Dataset for SoccerNet tracking data.

    __getitem__ returns:
        image:    Tensor[C, H, W]
        boxes:    Tensor[num_boxes, 4]  ([xmin, ymin, xmax, ymax])
        labels:   Tensor[num_boxes]     (all ones, or empty)
    """

    def __init__(self, img_size, sequence_dir):
        self.img_size = img_size
        self.sequence_dir = sequence_dir
        self.img_dir = os.path.join(sequence_dir, "img1")
        self.gt_txt = os.path.join(sequence_dir, "gt", "gt.txt")

        # build transform pipeline
        self.transform = T.Compose([
            T.Resize(self.img_size),
            T.ToTensor()
        ])

        # list + sort image files
        self.img_files = sorted(
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png"))
        )

        # parse annotation file once
        raw = np.loadtxt(self.gt_txt, delimiter=',')
        self.annotations = defaultdict(list)
        for row in raw:
            frame_id = int(row[0])
            x, y, w, h = row[2], row[3], row[4], row[5]
            xmin, ymin = float(x), float(y)
            xmax, ymax = xmin + float(w), ymin + float(h)
            self.annotations[frame_id].append([xmin, ymin, xmax, ymax])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # load + transform image
        img_name = self.img_files[idx]
        frame_idx = int(os.path.splitext(img_name)[0])
        path = os.path.join(self.img_dir, img_name)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        # fetch boxes + labels
        boxes = self.annotations.get(frame_idx, [])
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.ones(boxes.size(0), dtype=torch.int64)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)

        return img, boxes, labels

def collate_fn(batch):
    imgs, boxes_list, labels_list = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    boxes = pad_sequence(boxes_list, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-1)
    return imgs, boxes, labels

def get_dataloader(
    sequence_dir,
    img_size,
    batch_size=2,
    shuffle=False,
    num_workers=0
):
    """
    Returns a DataLoader over (images, gt_bboxes, gt_classes):
      images:    Tensor[B, C, H, W]
      gt_bboxes: Tensor[B, max_objs, 4]
      gt_classes:Tensor[B, max_objs]
    """
    dataset = SoccerDatasetSequence(img_size=img_size,
                                    sequence_dir=sequence_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )