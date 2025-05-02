import logging
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from captioner.dataset import make_mnist_captioning_dataset
from captioner.models import TransformerCaptioner
from captioner.utils import count_trainable_params, get_device
from captioner.utils.visualise import visualise_patched_input

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.INFO)

NUM_WORKERS = 1

def calculate_accuracy(scores, y, pad_token_id=None):
    # scores: B, N, N_classes
    # y: B, N
    if pad_token_id is not None:
        # Remove padding tokens from the target
        mask = y != pad_token_id
        scores = scores[mask]
        y = y[mask]
    correct = torch.sum(scores.argmax(dim=-1) == y)
    return correct/y.numel()

class Validator:
    def __init__(self, validation_dataloader: DataLoader, device: torch.device):
        self.valid_dl = validation_dataloader
        self.device = device
        self.pad_token_id = validation_dataloader.dataset.pad_token_id
        self.eos_token_id = validation_dataloader.dataset.eos_token_id

    def validate(self, model: torch.nn.Module, loss_fn: torch.nn.Module):
        total_correct = 0
        count = 0
        total_loss = 0.
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.valid_dl)):
                x, y = batch
                B, N_dec = y.shape 

                x = x.to(self.device)
                y = y.to(self.device)

                input = y.clone().detach()
                input[input == self.eos_token_id] = self.pad_token_id
                util_column = torch.full((B, 1), self.eos_token_id, 
                                        dtype=input.dtype, 
                                        device=input.device)
                input = torch.cat([util_column, input], dim=1)[:, :-1]
                
                target = y #torch.cat([y, util_column], dim=1)
                scores = model(x, input)

                # Then do the loss
                loss = loss_fn(scores.view(B*N_dec, -1), target.view(-1))
                total_loss += loss.item()

                mask = y != self.pad_token_id
                scores = scores[mask]
                target = target[mask]
                
                total_correct += torch.sum(scores.argmax(dim=-1) == target)
                count += torch.numel(target)

        return total_loss / len(self.valid_dl), total_correct / count

    def run_inference_single(self, model: torch.nn.Module, x):
        with torch.no_grad():
            x = x.to(self.device)
            scores = model(x, )
            # Get the predicted sequence
            pred_seq = scores.argmax(dim=-1)
            return pred_seq

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
            collate_fn=self.train_ds.collate_fn,
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=NUM_WORKERS,
            collate_fn=self.val_ds.collate_fn,
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
            # y is a padded sequence
            # do not add 1 for the (utility) <bos> or <eos> token
            # s.t. N_dec is the max length of the sequence in the batch

            x = x.to(self.device)
            y = y.to(self.device)
            # x shape B, D
            # decoder input is indexed until the <end> token

            # Scores are (unnormalised) logits
            input = y.clone().detach()
            input[input == self.train_dl.dataset.eos_token_id] = self.train_dl.dataset.pad_token_id
            util_column = torch.full((B, 1), 
                                     self.train_dl.dataset.eos_token_id, 
                                     dtype=input.dtype, 
                                     device=input.device)
            input = torch.cat([util_column, input], dim=1)[:, :-1]
            
            target = y #torch.cat([y, util_column], dim=1)
            scores = encoder(x, input)

            # Then do the loss
            loss = loss_fn(scores.view(B*N_dec, -1), target.view(-1))
            loss.backward()
            optimiser.step()

            # Gather data and report
            running_loss += loss.item()
            if batch_idx % batches_print_frequency == (batches_print_frequency - 1):
                logger.info(f'Correct seq:\t{",".join([str(i) for i in target[0].tolist()])}')
                logger.info(
                    f'Predicted seq:\t{",".join([str(i) for i in scores.argmax(-1)[0].tolist()])}')
                
                # Predict first token
                # visualise_patched_input(x_im.squeeze().cpu(), None, 16)
                with torch.no_grad():   
                    generated = input[0:1, :1]  
                    for _ in range(model.seq_len_dec - generated.size(1)):
                        test_scores = model(x[0:1], generated)    # assume outputs.logits [1, T, V]
                        # 3) Greedy pick at last position
                        next_token = torch.argmax(test_scores[:, -1, :], dim=-1, keepdim=True)  # [1,1]

                        # 4) Append and check EOS
                        generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]
                        if next_token.item() == self.train_dl.dataset.eos_token_id:
                            break
                    print(generated)       

                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimiser_state_dict': optimiser.state_dict(),
                    }

                    checkpoint_path = os.path.join(
                        '/Users/kenton/projects/mlx-institute/transformer/checkpoints',
                        f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth',
                    )
                    torch.save(checkpoint, checkpoint_path)  
                    
                

                # Calculate accuracy metric
                accuracy = calculate_accuracy(scores, target, pad_token_id=self.train_dl.dataset.pad_token_id)
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

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Example script with a --log_to_wandb flag"
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="If set, enable logging to Weights & Biases"
    )
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    log_to_wandb = args.log_to_wandb

    # Model configs
    model_config = {
        'patch_size': 16,
        'hidden_dim': 128,
        'num_heads': 8,
        'seq_len_enc': 196//4, # Number of patches 244/16 * 244/16 = 196
        'seq_len_dec': 17, # Number of tokens, fixed first
        'num_layers': 3,
        'dim_feedforward': 128,
        'num_classes': 12, # 10 digits + blank token + either <start> or <end>
    }
    # Config parameters
    setup_config = {'batch_size': 32}

    # Training configs
    training_config = {
        'epochs': 5,
        'lr': 1e-3,
        'log_locally': False,
        'log_to_wandb': log_to_wandb,
        'batches_print_frequency': 100,
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
                                                     patch_size=model_config['patch_size'],
                                                     return_empty_labels = False,
                                                     im_size = 112)
    
    optimiser = torch.optim.Adam(
        model.parameters(), lr=training_config.get('lr'),
    )

    # Load previously trained model
    # checkpoint_path = get_wandb_checkpoint_path(
    #     'kwokkenton-individual/mlx-week2-search-engine/towers_rnn:latest',
    # )

    # # Load the model
    checkpoint = torch.load(
        '/Users/kenton/projects/mlx-institute/transformer/checkpoints/20250502_010308.pth', 
        map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=train_ds.pad_token_id)

    trainer = Trainer(
        train_ds=train_ds,
        val_ds=val_ds,
        setup_config=setup_config,
        device=device,
    )
    # trainer.train_one_epoch(
    #     model=model,
    #     loss_fn=loss_fn,
    #     optimiser=optimiser,
    #     batches_print_frequency=training_config.get('batches_print_frequency'),
    # )

    trainer.train(
        epochs=training_config.get('epochs'),
        model=model,
        loss_fn=loss_fn,
        optimiser=optimiser,
        config=training_config,
    )
