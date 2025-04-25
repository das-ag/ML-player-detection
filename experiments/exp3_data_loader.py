import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

class FrameGTDataset(Dataset):
    """Dataset for per-frame images and ground truth annotations."""
    def __init__(self, sequence_dir, transforms=None):
        self.image_dir = os.path.join(sequence_dir, "img1")
        self.gt_dir = os.path.join(sequence_dir, "gt-frame")
        self.transforms = transforms

        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(f"Ground truth directory not found: {self.gt_dir}")

        # Gather samples: (image_path, gt_path, frame_id)
        self.samples = []
        for fname in sorted(os.listdir(self.image_dir)):
            if fname.lower().endswith((".jpg", ".png")):
                frame_id = os.path.splitext(fname)[0]
                img_path = os.path.join(self.image_dir, fname)
                gt_path = os.path.join(self.gt_dir, f"{frame_id}.txt")
                self.samples.append((img_path, gt_path, int(frame_id)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, gt_path, frame_id = self.samples[idx]
        image = read_image(img_path)
        image = convert_image_dtype(image, dtype=torch.float)

        # Load and parse ground truth for this frame
        boxes = []
        labels = []
        if os.path.exists(gt_path):
            with open(gt_path, "r") as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) != 5:
                        continue
                    track_id, ymin, xmin, ymax, xmax = map(int, parts)
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(track_id)

        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor([], dtype=torch.float32),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
            "orig_size": (image.shape[1], image.shape[2])
        }

        # Apply transforms and scale boxes if needed
        if self.transforms:
            orig_h, orig_w = target["orig_size"]
            image = self.transforms(image)
            new_h, new_w = image.shape[1], image.shape[2]
            if new_h != orig_h or new_w != orig_w:
                height_scale = new_h / orig_h
                width_scale = new_w / orig_w
                if boxes.shape[0] > 0:
                    boxes_scaled = target["boxes"].clone()
                    boxes_scaled[:, 0] *= width_scale
                    boxes_scaled[:, 2] *= width_scale
                    boxes_scaled[:, 1] *= height_scale
                    boxes_scaled[:, 3] *= height_scale
                    target["boxes"] = boxes_scaled
                    target["area"] = (boxes_scaled[:, 3] - boxes_scaled[:, 1]) * (boxes_scaled[:, 2] - boxes_scaled[:, 0])
                target["orig_size"] = (new_h, new_w)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataloader(sequence_dir, batch_size=4, shuffle=True, num_workers=4, transforms=None):
    """Helper to create DataLoader for a given sequence directory."""
    dataset = FrameGTDataset(sequence_dir, transforms=transforms)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return loader


if __name__ == "__main__":
    import sys
    from torchvision import transforms
    seq_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("..", "soccernet_data", "tracking", "train", "SNMOT-060")
    transform = transforms.Compose([transforms.Resize((800, 800))])
    loader = get_dataloader(seq_dir, batch_size=2, transforms=transform)
    for batch_idx, (images, targets) in enumerate(loader):
        print(f"Batch {batch_idx} ->", [img.shape for img in images], [t['boxes'].shape for t in targets])
        break 