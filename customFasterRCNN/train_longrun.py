import torch
import os
import numpy as np
import argparse
from data_loader import get_dataloader
from model import TwoStageDetector
from training_utils import training_loop, resume_training

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Player detection model training script')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--checkpoint-freq', type=int, default=20, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--data-dir', type=str, default="../soccernet_data/tracking/train/SNMOT-060", 
                        help='Path to training data directory')
    args = parser.parse_args()

    # Set up device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set image dimensions
    img_width = 1920
    img_height = 1080

    # Create checkpoint directory
    os.makedirs('checkpoints', exist_ok=True)

    # Get dataloader
    print(f"Loading data from {args.data_dir}")
    train_dataloader = get_dataloader(
        sequence_dir=args.data_dir, 
        img_size=(img_height, img_width), 
        batch_size=args.batch_size, 
        shuffle=True
    )

    # Initialize or load model
    if args.resume:
        # Load checkpoint to get model parameters
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        
        # Create model
        tmp_model = TwoStageDetector((img_height, img_width), 1024, 2, (2, 2))
        
        # Initialize feature extractor to examine output dimensions
        from featureExtractor import InceptionFeatureExtractor
        backbone = InceptionFeatureExtractor()
        batch = next(iter(train_dataloader))
        img_batch = batch[0].to(device)
        features = backbone(img_batch)
        
        # Get feature dimensions
        out_c, out_h, out_w = features.size(dim=1), features.size(dim=2), features.size(dim=3)
        
        # Create model with correct dimensions
        print(f"Creating model with output dimensions: {out_c}, {out_h}, {out_w}")
        model = TwoStageDetector((img_height, img_width), out_c, 2, (2, 2))
        
        # Load model weights from checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully")
        
        # Resume training
        print(f"Resuming training from epoch {checkpoint['epoch']} for {args.epochs} more epochs")
        loss_list = resume_training(
            model=model,
            train_dataloader=train_dataloader,
            device=device, 
            checkpoint_path=args.resume,
            n_epochs=args.epochs,
            save_checkpoint_freq=args.checkpoint_freq
        )
    else:
        # Create new model instance
        print("Initializing new model...")
        
        # Initialize feature extractor to examine output dimensions
        from featureExtractor import InceptionFeatureExtractor
        backbone = InceptionFeatureExtractor()
        batch = next(iter(train_dataloader))
        img_batch = batch[0].to(device)
        features = backbone(img_batch)
        
        # Get feature dimensions
        out_c, out_h, out_w = features.size(dim=1), features.size(dim=2), features.size(dim=3)
        
        # Create model with correct dimensions
        print(f"Creating model with output dimensions: {out_c}, {out_h}, {out_w}")
        model = TwoStageDetector((img_height, img_width), out_c, 2, (2, 2))
        
        # Start training from scratch
        print(f"Starting training for {args.epochs} epochs")
        loss_list = training_loop(
            model=model,
            learning_rate=args.lr,
            train_dataloader=train_dataloader,
            n_epochs=args.epochs,
            device=device,
            save_checkpoint_freq=args.checkpoint_freq
        )

    # Save losses to file
    np.save('checkpoints/training_losses.npy', np.array(loss_list))
    print("Training complete!")
    print(f"Final loss: {loss_list[-1]:.4f}")
    print("Loss history saved to checkpoints/training_losses.npy")

if __name__ == "__main__":
    main() 