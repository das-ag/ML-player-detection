import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype

class SoccerNetMOTDataset(Dataset):
    def __init__(self, sequence_dir, transforms=None):
        """
        Args:
            sequence_dir (string): Path to a specific SNMOT sequence folder (e.g., 'soccernet_data/tracking/train/SNMOT-060').
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.sequence_dir = sequence_dir
        self.img_dir = os.path.join(sequence_dir, "img1")
        self.gt_file = os.path.join(sequence_dir, "gt", "gt.txt")
        self.transforms = transforms
        self.img_paths = []
        self.annotations = self._load_annotations()

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Store frame IDs along with paths for easier annotation lookup
        for img_name in sorted(os.listdir(self.img_dir)):
            if img_name.endswith(".jpg"):
                try:
                    # Frame IDs are 1-based in gt.txt, filenames are 6-digit 0-padded
                    frame_id = int(img_name.split('.')[0])
                    self.img_paths.append((os.path.join(self.img_dir, img_name), frame_id))
                except ValueError:
                    print(f"Warning: Could not parse frame ID from image filename: {img_name}")

    def _load_annotations(self):
        if not os.path.exists(self.gt_file):
            print(f"Error: Ground truth file not found at {self.gt_file}")
            return pd.DataFrame() # Return empty dataframe

        try:
            # Columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            df = pd.read_csv(self.gt_file, header=None, index_col=False)
            df.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf", "x", "y", "z"]
            df = df[df['conf'] == 1] # Keep only actual detections (ignore placeholders)
            # Convert to PyTorch expected format [xmin, ymin, xmax, ymax]
            # Perform calculations directly for efficiency
            df['xmin'] = df['bb_left']
            df['ymin'] = df['bb_top']
            df['xmax'] = df['bb_left'] + df['bb_width']
            df['ymax'] = df['bb_top'] + df['bb_height']
            # Set frame as index for potentially faster lookup in __getitem__
            # df = df.set_index('frame')
            return df
        except Exception as e:
            print(f"Error reading or processing ground truth file {self.gt_file}: {e}")
            return pd.DataFrame()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, frame_id = self.img_paths[idx]
        image = read_image(img_path)
        image = convert_image_dtype(image, dtype=torch.float)
        
        # Store original dimensions for proper box scaling
        orig_height, orig_width = image.shape[1], image.shape[2]

        # Get annotations for this specific frame
        # Ensure annotations DataFrame is not empty
        if self.annotations.empty:
             # Handle case with no annotations gracefully (e.g., return image with empty targets)
             print(f"Warning: Annotations are empty. Returning image {idx} with no boxes.")
             boxes = torch.empty((0, 4), dtype=torch.float32)
             labels = torch.empty((0,), dtype=torch.int64)
        else:
            frame_annotations = self.annotations[self.annotations['frame'] == frame_id]
            boxes = frame_annotations[['xmin', 'ymin', 'xmax', 'ymax']].values
            # Check if boxes were actually found for this frame_id
            if boxes.shape[0] == 0:
                # print(f"Warning: No annotations found for frame {frame_id} in {self.gt_file}. Returning empty boxes.")
                boxes = torch.empty((0, 4), dtype=torch.float32)
                labels = torch.empty((0,), dtype=torch.int64)
            else:
                # For this task, all objects are 'player', so label is 1 (0 is background)
                labels = torch.ones((len(boxes),), dtype=torch.int64)

        # Create target dict
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx]) # Use dataset index as image_id
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0]) if boxes.shape[0] > 0 else torch.empty((0,), dtype=torch.float32)
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Store original dimensions for scaling reference
        target["orig_size"] = (orig_height, orig_width)

        if self.transforms:
            # Apply transforms to the image
            image = self.transforms(image)
            
            # Scale boxes if image size changed
            if image.shape[1] != orig_height or image.shape[2] != orig_width:
                new_height, new_width = image.shape[1], image.shape[2]
                # Scale factor for each dimension
                height_scale = new_height / orig_height
                width_scale = new_width / orig_width
                
                # Only scale if there are boxes
                if len(target["boxes"]) > 0:
                    # Scale boxes: [x1, y1, x2, y2]
                    boxes_scaled = target["boxes"].clone()
                    boxes_scaled[:, 0] *= width_scale  # x1
                    boxes_scaled[:, 2] *= width_scale  # x2
                    boxes_scaled[:, 1] *= height_scale  # y1
                    boxes_scaled[:, 3] *= height_scale  # y3
                    target["boxes"] = boxes_scaled
                    # Recalculate area
                    target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
                    
        return image, target

if __name__ == '__main__':
    # Example Usage:
    # Update example to use a specific sequence directory
    sequence_directory = "soccernet_data/tracking/train/SNMOT-060" 
    print(f"Running Dataset example for sequence: {sequence_directory}")
    
    if not os.path.exists(sequence_directory):
        print(f"Error: Sequence directory not found at {sequence_directory}")
        print("Skipping Dataset example.")
    else:
        # Basic transform (more sophisticated ones needed for training)
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((800, 800)) # Example resize, adjust as needed
        ])

        dataset = SoccerNetMOTDataset(sequence_dir=sequence_directory, transforms=None) # No transform initially
        print(f"Dataset size: {len(dataset)}")

        if len(dataset) > 0:
            # Check a few samples
            for i in range(min(3, len(dataset))):
                img, target = dataset[i] # Get a sample
                print(f"\nSample {i}:")
                print(f"  Image shape: {img.shape}")
                print(f"  Target keys: {target.keys()}")
                print(f"  Number of boxes: {len(target['boxes'])}")
                if len(target['boxes']) > 0:
                    print(f"  First box: {target['boxes'][0]}")
                    print(f"  First label: {target['labels'][0]}")
                else:
                    print("  No boxes in this sample.")

            # Example with transform
            dataset_transformed = SoccerNetMOTDataset(sequence_dir=sequence_directory, transforms=transform)
            if len(dataset_transformed) > 0:
                img_t, target_t = dataset_transformed[0]
                print(f"\nTransformed Image shape (Sample 0): {img_t.shape}")
                print(f"Transformed Target boxes (Sample 0): {len(target_t['boxes'])}")
            else:
                print("Transformed dataset is empty.")
        else:
            print("Dataset is empty. Check sequence_dir and gt.txt path within it.") 