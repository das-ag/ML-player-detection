import os
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

# ensure project root is in path for dataset import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import SoccerNetMOTDataset

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

if __name__ == "__main__":
    # Adjust this sequence ID as needed
    train_root = os.path.join("..", "soccernet_data", "tracking", "train")
    sequence_id = "SNMOT-060"
    sequence_dir = os.path.join(train_root, sequence_id)
    print(f"Building DataLoader for sequence: {sequence_dir}")

    # Example transform: resize images (adjust to match your model requirements)
    transform = transforms.Compose([
        transforms.Resize((800, 1333)),
    ])

    # Initialize dataset and DataLoader
    dataset = SoccerNetMOTDataset(sequence_dir=sequence_dir, transforms=transform)
    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )

    # Iterate through a few batches to verify
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        print(f"  Number of images: {len(images)}")
        print(f"  Example image shape: {images[0].shape}")
        print(f"  Number of boxes in first image: {targets[0]['boxes'].shape[0]}")
        if batch_idx == 2:
            break 