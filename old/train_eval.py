import wandb
import torch

def train_model(model, train_loader, test_loader, optimizer, criterion, device, num_epochs=10, run_name="run_01"):
    """
    Train the model, logging metrics to W&B.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
        optimizer: The optimizer to use (e.g., Adam)
        criterion: The loss function (e.g., CrossEntropyLoss)
        device: The device to train the model on ('cuda' or 'cpu')
        num_epochs: The number of epochs to train for
        run_name: The name of the W&B run
    """
    model.to(device)

    # Initialize W&B logging
    wandb.init(entity="emilengdahl", project="ImageTransformer", config={
        "epochs": num_epochs,
        "batch_size": train_loader.batch_size,
        "model": "ImageTransformer",  # Adjust this to match your model's name
        "optimizer": "Adam",
        "lr": optimizer.param_groups[0]['lr']
    }, name=run_name)

    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for patches, labels in train_loader:
            patches = patches.to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(patches)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Calculate training accuracy
        acc = correct / total

        # Log training loss and accuracy to W&B
        wandb.log({
            "train_loss": total_loss / len(train_loader),
            "train_accuracy": acc,
            "epoch": epoch + 1
        })

        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}")

        # Evaluate the model after every epoch
        eval_acc = evaluate_model(model, test_loader, device)
        
        # Log evaluation accuracy
        wandb.log({
            "test_accuracy": eval_acc,
            "epoch": epoch + 1
        })

    # Finish W&B run
    wandb.finish()


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on the test set and log the accuracy.
    
    Args:
        model: The trained model
        test_loader: DataLoader for testing data
        device: The device to evaluate the model on ('cuda' or 'cpu')
    
    Returns:
        acc: The test accuracy
    """
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for patches, labels in test_loader:
            patches = patches.to(device).float()
            labels = labels.to(device)

            outputs = model(patches)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Calculate accuracy
    acc = correct / total
    print(f"Test Accuracy: {acc:.4f}")
    return acc

