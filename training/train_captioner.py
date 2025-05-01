import logging
import os
from datetime import datetime

import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioner.dataset import make_mnist_captioning_dataset
from captioner.models import TransformerCaptioner
from captioner.utils import count_trainable_params, get_device

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

NUM_WORKERS = 4

def calculate_accuracy(scores, y):
    # scores: B, N, N_classes
    # y: B, N
    correct = torch.sum(scores.argmax(dim=-1) == y)
    return correct/y.numel()

class Validator:
    def __init__(self, validation_dataloader: DataLoader, device: torch.device):
        self.valid_dl = validation_dataloader
        self.device = device

    def validate(self, model: torch.nn.Module, loss_fn: torch.nn.Module, score_fn):
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
        train_ds,
        val_ds,
        setup_config: dict,
        device: torch.device,
    ):

        self.setup_config = setup_config
        self.batch_size = setup_config.get('batch_size')
        self.device = device
        self.train_ds = train_ds
        self.val_ds = val_ds
        # Set up datasets and dataloaders

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
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        batches_print_frequency: int = 100,
    ):

        running_loss = 0.
        last_loss = 0.

        encoder = model.to(self.device)

        for batch_idx, batch in enumerate(tqdm(self.train_dl)):
            # Zero your gradients for every batch!
            optimiser.zero_grad()
            x, y = batch
            B, N_dec = y.shape 
            # y is double padded by default
            N_dec -= 1

            x = x.to(self.device)
            y = y.to(self.device)
            # x shape B, D
            # decoder input is indexed until the <end> token
            # Scores are (unnormalised) logits

            scores = encoder(x, y[..., :-1])

            # Then do the loss
            # target has the <start> token removed
            # Do some dirty copying to shift the token indices
            target = y[...,1:].clone().detach()
            target[..., -1] = y[..., 0].clone().detach()

            loss = loss_fn(scores.view(B*N_dec, -1), target.view(-1))
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):
                logger.info(f'Correct seq:\t{",".join([str(i) for i in target[0].tolist()])}')
                logger.info(
                    f'Predicted seq:\t{",".join([str(i) for i in scores.argmax(-1)[0].tolist()])}')
                 
                # Calculate accuracy metric
                accuracy = calculate_accuracy(scores, target)
                ppl = torch.exp(loss)
                # loss per batch
                last_loss = running_loss / batches_print_frequency
                logger.info(
                    f'  For batch {batch_idx + 1}, the loss is {last_loss}, the accuracy is {accuracy}, the perplexity is {ppl}, ',
                )
                running_loss = 0.

        return last_loss, accuracy


    
    def train(
        self,
        epochs: int,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        config: dict,
    ):

        logger.info(f'Training config: {config}')
        log_to_wandb = config.get('log_to_wandb')
        log_locally = config.get('log_locally')
        checkpoint_folder = config.get(
            'checkpoint_folder',
            '/Users/kenton/projects/mlx-institute/transformer/checkpoints',
        )

        if log_to_wandb:
            run = wandb.init(
                entity='kwokkenton-individual',
                project='mlx-week3-image-captioning',
                config=config,
            )

        for epoch in range(epochs):
            logger.info(f'Training: Epoch {epoch + 1} of {epochs}')
            train_loss, train_accuracy = self.train_one_epoch(
                model, loss_fn, optimiser, config.get(
                    'batches_print_frequency',
                ),
            )
            logger.info(
                f'Validating: Epoch {epoch + 1} of {epochs}.',
            )
            # Run validation to sanity check the model
            val_loss, val_accuracy = self.validator.validate(
                model, loss_fn,
            )
            logger.info(
                f'Epoch {epoch + 1} of {epochs} train loss: {train_loss}'
                f'train accuracy: {train_accuracy} val loss: {val_loss}'
                f'val accuracy: {val_accuracy}',
            )
            if log_locally or log_to_wandb:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
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
    
    # Model configs
    model_config = {
        'patch_size': 16,
        'hidden_dim': 256,
        'num_heads': 12,
        'seq_len_enc': 196, # Number of patches 244/16 * 244/16 = 196
        'seq_len_dec': 65, # Number of tokens, fixed first
        'num_layers': 6,
        'dim_feedforward': 512,
        'num_classes': 12, # 10 digits + blank token + either <start> or <end>
    }
    # Config parameters
    setup_config = {'batch_size': 2}

    # Training configs
    training_config = {
        'epochs': 5,
        'lr': 1e-3,
        'log_locally': True,
        'log_to_wandb': False,
        'batches_print_frequency': 10,
    }

    device = get_device()

    model = TransformerCaptioner(
        in_dim_enc = model_config['patch_size'] * model_config['patch_size'],
        d_model=model_config['hidden_dim'],
        num_heads = model_config['num_heads'],
        seq_len_enc=model_config['seq_len_enc'],
        seq_len_dec=model_config['seq_len_dec'],
        num_layers = model_config['num_layers'],
        dim_feedforward=model_config['dim_feedforward'],
        num_classes=model_config['num_classes'],
    )
    num_params = count_trainable_params(model)
    logger.info(f'There are {num_params} trainable parameters in the model.')
    logger.info(model)

    train_ds, val_ds = make_mnist_captioning_dataset('~/data', 
                                                     patch = True, 
                                                     patch_size=model_config['patch_size'])
    
    optimiser = torch.optim.Adam(
        model.parameters(), lr=training_config.get('lr'),
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
        train_ds=train_ds,
        val_ds=val_ds,
        setup_config=setup_config,
        device=device,
    )
    trainer.train_one_epoch(
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        batches_print_frequency=training_config.get('batches_print_frequency'),
    )

    # trainer.train(
    #     epochs=training_config.get('epochs'),
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimiser=optimiser,
    #     config=training_config,
    # )
