import logging
import os
import random
from datetime import datetime
from functools import partial

import torch
import torchvision
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioner.dataset import MNIST_PATH, to_tensor
from captioner.models import CNN_Encoder
from captioner.utils import get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

NUM_WORKERS = 1


class Validator:
    def __init__(self, validation_dataloader: DataLoader, device: torch.device):
        self.valid_dl = validation_dataloader
        self.device = device

    def validate(self, model: torch.nn.Module, loss_fn: torch.nn.Module):
        total_correct = 0
        count = 0
        total_loss = 0.
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_dl)):
                x, y = batch
                x = x.to(self.device)   
                y = y.to(self.device)

                scores = model(x)
                loss = loss_fn(scores, y)
                total_loss += loss.item()
                correct = torch.sum(scores.argmax(dim=1) == y) 
                total_correct += correct
                count += len(y)

        return total_loss / len(self.valid_dl), total_correct / count

class Trainer:
    def __init__(
        self, 
        setup_config: dict,
        device: torch.device,
    ):

        self.setup_config = setup_config
        self.batch_size = setup_config.get('batch_size')
        self.device = device

        # Set up datasets and dataloaders
        self.train_ds = torchvision.datasets.MNIST(MNIST_PATH, train=True, download=True, transform=to_tensor)
        self.val_ds = torchvision.datasets.MNIST(MNIST_PATH, train=False, download=True, transform=to_tensor)

        self.train_dl = DataLoader(
            self.train_ds, 
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )
        self.val_dl = DataLoader(
            self.val_ds, 
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
        )

        self.validator = Validator(self.val_dl, self.device)

    def train_one_epoch(
        self,
        models: dict[str, torch.nn.Module],
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 100,
    ):

        running_loss = 0.
        last_loss = 0.

        encoder = models.get('encoder').to(self.device)

        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            # Zero your gradients for every batch!
            optimiser.zero_grad()
    
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            scores = encoder(x)
   
            # Then do the loss
            loss = loss_fn(scores, y)
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):
                # Calculate accuracy metric
                accuracy = torch.sum(scores.argmax(dim=1) == y) / len(y)
                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  For batch {batch_idx + 1}, the loss is {last_loss} and the accuracy is {accuracy}',
                )
                running_loss = 0.

        return last_loss, accuracy

    def train(
        self,
        epochs: int,
        models: dict[str, torch.nn.Module],
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        config: dict,
    ):

        logger.info(f'Training config: {config}')
        log_to_wandb = config.get('log_to_wandb')
        log_locally = config.get('log_locally')
        checkpoint_folder = config.get('checkpoint_folder', 
                                     '/Users/kenton/projects/mlx-institute/transformer/checkpoints')

        if log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project='mlx-week3-image-captioning',
                config=config,
            )

        for epoch in range(epochs):
            logger.info(f'Training: Epoch {epoch + 1} of {epochs}')
            train_loss, train_accuracy = self.train_one_epoch(
                models, loss_fn, optimiser, config.get('batches_print_frequency'),
            )
            logger.info(
                f'Validating: Epoch {epoch + 1} of {epochs}.',
            )
            # Run validation to sanity check the model
            val_loss, val_accuracy = self.validator.validate(
                models['encoder'], loss_fn,
            )
            logger.info(
                f'Epoch {epoch + 1} of {epochs} train loss: {train_loss}'
                f'train accuracy: {train_accuracy} val loss: {val_loss}'
                f'val accuracy: {val_accuracy}',
            )
            if log_locally or log_to_wandb:
                checkpoint = {
                    'encoder_state_dict': models['encoder'].state_dict(),
                    'optimiser_state_dict': optimiser.state_dict(),
                }
                
                if log_locally:
                    checkpoint_path = os.path.join(
                        checkpoint_folder, 
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                    )
                    torch.save(checkpoint, checkpoint_path)

                if log_to_wandb:
                    wandb.log({
                        'train/loss': train_loss,
                        'train/accuracy': train_accuracy,
                        'val/loss': val_loss,
                        'val/accuracy': val_accuracy,
                    })

                    checkpoint_path = os.path.join(
                        wandb.run.dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                    )
                    torch.save(checkpoint, checkpoint_path)
                    artifact = wandb.Artifact('cnn_encoder', type='checkpoint')
                    artifact.add_file(checkpoint_path)
                    wandb.run.log_artifact(artifact)


if __name__ == '__main__':
    # Training configs
    # Model configs

    # Config parameters
    setup_config = {'batch_size': 32}
    
    training_config = {
        'epochs': 5,
        'lr': 1e-3,
        'log_locally': True,
        'log_to_wandb': False,
        'batches_print_frequency': 500,
    }

    device = get_device()

    models = {
        'encoder': CNN_Encoder(),
    }

    optimiser = torch.optim.Adam(
        list(models['encoder'].parameters()), lr=training_config.get('lr'),
    )

    # Load previously trained model
    # checkpoint_path = get_wandb_checkpoint_path(
    #     'kwokkenton-individual/mlx-week2-search-engine/towers_rnn:latest',
    # )

    # # Load the model
    # checkpoint = torch.load(
    #     checkpoint_path, map_location=device, weights_only=True,
    # )
    # query_encoder.load_state_dict(checkpoint['query_encoder_state_dict'])
    # doc_encoder.load_state_dict(checkpoint['doc_encoder_state_dict'])

    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(
        setup_config=setup_config,
        device=device,
    )

    trainer.train(
        epochs=training_config.get('epochs'),
        models=models,
        loss_fn=loss_fn,
        optimiser=optimiser,
        config=training_config,
    )
