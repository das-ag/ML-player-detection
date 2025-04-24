import torch
from tqdm import tqdm
import torch.optim as optim
from torch.amp import GradScaler, autocast # For mixed precision
import os
from datetime import datetime

def training_loop(model, learning_rate, train_dataloader, n_epochs, device, 
                  save_checkpoint_freq=10, checkpoint_dir='checkpoints'):
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler - CosineAnnealing with warm restarts
    # Suitable for very long training (1000 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,          # Initial restart interval (epochs)
        T_mult=2,        # Multiplier for restart intervals
        eta_min=1e-6     # Minimum learning rate
    )
    
    scaler = GradScaler(device=device.type) 

    model.to(device) # Move model to the selected device
    model.train()
    loss_list = []
    best_loss = float('inf')

    for i in range(n_epochs):
        print(f"Epoch {i+1}/{n_epochs}")
        total_loss = 0
        loop = tqdm(train_dataloader, leave=True)
        for img_batch, gt_bboxes_batch, gt_classes_batch in loop:

            # Move data to device
            img_batch = img_batch.to(device)
            gt_bboxes_batch = gt_bboxes_batch.to(device)
            # ***** Ensure gt_classes are LongTensor and moved to device *****
            # Assuming gt_classes_batch is a tensor from the dataloader
            gt_classes_batch = gt_classes_batch.long().to(device)


            optimizer.zero_grad(set_to_none=True) # Use set_to_none=True for potential minor performance gain

            with autocast(device_type=device.type):
                loss = model(img_batch, gt_bboxes_batch, gt_classes_batch, device=device)

            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Update tqdm progress bar description
            loop.set_description(f"Epoch {i+1}")
            loop.set_postfix(loss=batch_loss, lr=optimizer.param_groups[0]['lr'])

        avg_epoch_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_epoch_loss)
        print(f"Epoch {i+1} Average Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Save checkpoints based on frequency
        if (i+1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{i+1}.pt")
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {i+1}: {checkpoint_path}")
        
        # Always save the best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"final_model_epoch_{n_epochs}.pt")
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_loss,
    }, final_path)
    print(f"Final model saved: {final_path}")

    return loss_list

def resume_training(model, train_dataloader, device, checkpoint_path, 
                   n_epochs=1000, save_checkpoint_freq=10, checkpoint_dir='checkpoints'):
    """
    Resume training from a checkpoint
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        device: Device to train on
        checkpoint_path: Path to the checkpoint to resume from
        n_epochs: Total number of epochs to train (including previous)
        save_checkpoint_freq: How often to save checkpoints (in epochs)
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        List of losses for each epoch
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device FIRST
    model.to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer (required even if loading from checkpoint)
    optimizer = optim.Adam(model.parameters())
    
    # Load optimizer state and make sure it's on the right device
    optimizer_state = checkpoint['optimizer_state_dict']
    
    # Fix optimizer state device
    for state in optimizer_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    optimizer.load_state_dict(optimizer_state)
    
    # Create scheduler and load state
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get the starting epoch
    start_epoch = checkpoint['epoch']
    
    # Get best loss so far if available
    best_loss = checkpoint.get('loss', float('inf'))
    
    scaler = GradScaler(device=device.type)
    
    model.train()
    loss_list = []
    
    print(f"Resuming training from epoch {start_epoch}")
    
    for i in range(start_epoch, n_epochs):
        print(f"Epoch {i+1}/{n_epochs}")
        total_loss = 0
        loop = tqdm(train_dataloader, leave=True)
        for img_batch, gt_bboxes_batch, gt_classes_batch in loop:
            # Move data to device
            img_batch = img_batch.to(device)
            gt_bboxes_batch = gt_bboxes_batch.to(device)
            gt_classes_batch = gt_classes_batch.long().to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type):
                loss = model(img_batch, gt_bboxes_batch, gt_classes_batch, device=device)

            if torch.isnan(loss):
                print("Warning: NaN loss detected. Skipping batch.")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Scale loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_loss = loss.item()
            total_loss += batch_loss

            # Update tqdm progress bar description
            loop.set_description(f"Epoch {i+1}")
            loop.set_postfix(loss=batch_loss, lr=optimizer.param_groups[0]['lr'])

        avg_epoch_loss = total_loss / len(train_dataloader)
        loss_list.append(avg_epoch_loss)
        print(f"Epoch {i+1} Average Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.8f}")
        
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Save checkpoints based on frequency
        if (i+1) % save_checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{i+1}.pt")
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {i+1}: {checkpoint_path}")
        
        # Always save the best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': i+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"final_model_epoch_{n_epochs}.pt")
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_loss,
    }, final_path)
    print(f"Final model saved: {final_path}")

    return loss_list 