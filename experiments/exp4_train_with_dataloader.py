# experiments/exp4_train_with_dataloader.py
import os
import torch
import torch.optim as optim
import torchvision.transforms as T
from exp3_data_loader import get_dataloader

# 1) Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Hyperparameters and data paths
sequence_dir = os.path.join("..", "soccernet_data", "tracking", "train", "SNMOT-060")
batch_size = 4
num_workers = 4
n_epochs = 20
lr = 1e-4

# 3) Image transforms
y_transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# 4) Initialize DataLoader
loader = get_dataloader(
    sequence_dir,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    transforms=y_transform
)

# 5) Prepare the RPN (assumes `rpn` is already defined in the environment)
rpn.to(device)
# Freeze backbone parameters
for p in rpn.feature_extractor.parameters():
    p.requires_grad = False

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, rpn.parameters()),
    lr=lr
)

# 6) Training loop
for epoch in range(1, n_epochs + 1):
    rpn.train()
    epoch_loss = 0.0
    for images, targets in loader:
        # Move images to device
        images = [img.to(device) for img in images]

        # Pad targets in batch to same number of boxes
        max_boxes = max([t["boxes"].shape[0] for t in targets])
        gt_boxes = []
        gt_labels = []
        for t in targets:
            b, l = t["boxes"], t["labels"]
            n = b.shape[0]
            if n < max_boxes:
                pad_b = torch.zeros((max_boxes - n, 4), device=device)
                pad_l = torch.zeros((max_boxes - n,), dtype=torch.int64, device=device)
                b = torch.cat([b.to(device), pad_b], dim=0)
                l = torch.cat([l.to(device), pad_l], dim=0)
            else:
                b = b.to(device)
                l = l.to(device)
            gt_boxes.append(b)
            gt_labels.append(l)
        # Stack into tensors of shape [B, N, ...]
        gt_boxes = torch.stack(gt_boxes, dim=0)
        gt_labels = torch.stack(gt_labels, dim=0)

        # Forward + backward
        optimizer.zero_grad()
        loss, _, _, _, _ = rpn(torch.stack(images, dim=0), gt_boxes, gt_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(loader)
    print(f"[Epoch {epoch}/{n_epochs}] RPN loss = {avg_loss:.4f}") 