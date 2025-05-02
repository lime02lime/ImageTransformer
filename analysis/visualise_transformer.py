import torch

from captioner.dataset import (
    label_tensor_to_string,
    make_mnist_captioning_dataset,
    patchify,
    unpatch,
)
from captioner.dataset.dataset import pred_to_string
from captioner.models import TransformerCaptioner
from captioner.utils import get_device
from captioner.utils.visualise import visualise_patched_input

IM_SIZE = 112

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
device = get_device()
checkpoint = torch.load(
    '/Users/kenton/projects/mlx-institute/transformer/checkpoints/20250502_023824.pth', 
    map_location=device, weights_only=True,
)
model.load_state_dict(checkpoint['model_state_dict'])

train_ds, val_ds = make_mnist_captioning_dataset('~/data', 
                                                    patch=True, 
                                                    patch_size=model_config['patch_size'], 
                                                    return_empty_labels=False)
X, y = val_ds[0]
B = 1
# visualise_patched_input(X, pred_to_string(y, True), model_config['patch_size'], )

dec_input = y.clone().detach().unsqueeze(0)  
dec_input[dec_input == val_ds.eos_token_id] = val_ds.pad_token_id
util_column = torch.full((B, 1), 
                            val_ds.pad_token_id, 
                            dtype=dec_input.dtype, 
                            device=dec_input.device)
dec_input = torch.cat([util_column, dec_input], dim=1)[:, :-1]

test_scores = model(X.unsqueeze(0)[0:1], dec_input)    # assume outputs.logits [1, T, V]
