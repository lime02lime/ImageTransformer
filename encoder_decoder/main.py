from datetime import datetime
import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
from model import CustomTransformerEncoderDecoder
from data_prep import load_dataset_and_loader, create_and_save_datasets
import os


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (patches, labels) in enumerate(progress_bar):
        # Move data to device
        patches = patches.to(device)  # Shape: [B, N, patch_dim]
        labels = labels.to(device)    # Shape: [B, seq_len]
        
        # Prepare input for decoder (shift right)
        tgt_input = labels[:, :-1]  # Remove last token
        tgt_output = labels[:, 1:]   # Remove first token (start token)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(patches, tgt_input)  # Shape: [B, seq_len-1, num_classes]
        
        # Calculate loss
        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        current_loss = total_loss / (batch_idx + 1)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})
        
        # Log to wandb
        wandb.log({
            'batch_loss': loss.item(),
            'running_loss': current_loss,
            'epoch': epoch,
            'batch': batch_idx
        })
    
    epoch_loss = total_loss / num_batches
    return epoch_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for patches, labels in val_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            
            # Prepare input for decoder
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]
            
            # Forward pass
            output = model(patches, tgt_input)
            
            # Calculate loss
            loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            correct_predictions += (predictions == tgt_output).sum().item()
            total_predictions += tgt_output.numel()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, config, save_path='best_model.pth'):
    """
    Main training function with wandb integration
    
    Args:
        model: The transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Dictionary containing training configuration
    """

    timestamp = datetime.now().strftime("%H:%M:%S")
    run_name = f"run-{timestamp}"

    # Initialize wandb
    wandb.init(
        entity="emilengdahl",
        project="ImageTransformer",
        config=config,
        name=run_name, reinit=True, id=None
    )
    
    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=12)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config['min_lr']
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(1, config['epochs']+1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': current_lr
        })
        
        # Save best model
        if val_loss < best_val_loss:

            # Save model checkpoint to file
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)

            best_val_loss = val_loss
            artifact = wandb.Artifact('best-model', type='model')
            artifact.add_file('best_model.pth')
            wandb.log_artifact(artifact)
        
        print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}')
    
    wandb.finish()



import matplotlib.pyplot as plt

def show_test_examples(model, test_loader, config, device, num_examples=5):
    model.eval()
    start_token = 10  # START token index
    stop_token = 11   # STOP token index
    with torch.no_grad():
        shown = 0
        for patches, labels in test_loader:
            patches = patches.to(device)
            labels = labels.to(device)
            for i in range(patches.size(0)):
                patch = patches[i].unsqueeze(0)  # (1, num_patches, patch_dim)
                label = labels[i]
                label = label[(label != 10) & (label != 11) & (label != 12)]  # Remove start, stop, and padding tokens
                # Generate output
                generated = model.generate(
                    patch, start_token=start_token, stop_token=stop_token,
                    device=device, max_len=config['max_seq_length']
                )
                # Plot the image
                # Reconstruct the image from patches for visualization
                img = patch_to_image(patch.cpu().squeeze(0), config)
                plt.figure(figsize=(4,4))
                plt.imshow(img, cmap='gray')
                plt.axis('off')
                plt.title(f"  Act: {label.tolist()}\nPred: {generated}")
                
                plt.savefig(f"example_{shown}.png")  # save instead of show
                #plt.show()
                shown += 1
                if shown >= num_examples:
                    print(f"{num_examples} examples generated and saved.")
                    return

def patch_to_image(patches, config):
    # Reconstructs the image from patches for visualization
    patch_size = int(config['patch_dim'] ** 0.5)
    grid_size = int(config['num_patches'] ** 0.5)
    img = torch.zeros(grid_size * patch_size, grid_size * patch_size)
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            patch = patches[idx].reshape(patch_size, patch_size)
            img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch
            idx += 1
    return img




def main():

    train_or_eval = input("Train or eval? (train/eval): ").strip().lower()

    train_path = 'encoder_decoder/train_file.pt'
    test_path = 'encoder_decoder/test_file.pt'

    # Check if data files exist, otherwise create them
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("Data files not found. Creating datasets...")
        try:
            create_and_save_datasets()
        except Exception as e:
            print(f"Error creating datasets: {e}")
            exit(1)
    else:
        print("Data files found. Skipping dataset creation.")
    
    # config for training
    config = {
        'learning_rate': 1e-3,
        'min_lr': 5e-7,
        'weight_decay': 0.01,
        'epochs': 25,
        'batch_size': 512,
        'num_heads': 8,
        'emb_dim': 2048,
        'ff_hidden_dim': 256,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'patch_dim': 196, 
        'num_classes': 13,  # 10 digits + start/stop tokens
        'max_seq_length': 12,
        'num_patches': 64   
    }

    # Load data
    try:
        train_loader = load_dataset_and_loader(train_path, batch_size=config['batch_size'])
        test_loader = load_dataset_and_loader(test_path, batch_size=config['batch_size'])
    except Exception as e:
        print(f"Error loading datasets: {e}")
        exit(1)

    print("Data loaded successfully. Initiating model...")

    model = CustomTransformerEncoderDecoder(
        patch_dim=config['patch_dim'],        # 196
        emb_dim=config['emb_dim'],           # 256
        num_heads=config['num_heads'],        # 8
        ff_hidden_dim=config['ff_hidden_dim'],# 512
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        num_classes=config['num_classes'],    # 13
        num_patches=config['num_patches'],    # 64 (matches your input)
        max_seq_length=config['max_seq_length'] # 12 (matches label sequence length)
    )
    
    # Your training loop
    model_save_path = 'best_model_768D.pth'
    
    if train_or_eval == 'train':
        # Train the model
        print("Training the model...")
        train_model(model, train_loader, test_loader, config, save_path=model_save_path)
        print("Training completed. Model saved.")
    elif train_or_eval == 'eval':
        # Check if the model file exists
        if not os.path.exists(model_save_path):
            print(f"Model file {model_save_path} not found. Please train the model first.")
            exit(1)
        else:
            print(f"Model file {model_save_path} found. Proceeding with evaluation.")

    # Show some test examples
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_save_path)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # in this case we save these example outputs as images, so this works both locally and through remote
    print("Evaluating and saving test examples...")
    show_test_examples(model, test_loader, config, device, num_examples=5)


if __name__ == "__main__":
    main()
