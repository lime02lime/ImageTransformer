import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas

from captioner.dataset import (
    label_tensor_to_string,
    make_mnist_captioning_dataset,
    patchify,
    unpatch,
)
from captioner.models import TransformerCaptioner
from captioner.utils import get_device

# PATCH_SIZE = 16
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

@st.cache_resource
def load_mnist(data, patch, patch_size):
    train_ds, val_ds = make_mnist_captioning_dataset(data, 
                                                     patch=patch, 
                                                     patch_size=patch_size, 
                                                     return_empty_labels=False)
    return train_ds, val_ds

@st.cache_resource
def prepare_model():
    """
    Load the model and weights
    """
    # Load the model
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
    checkpoint = torch.load(
    '/Users/kenton/projects/mlx-institute/transformer/checkpoints/20250502_023824.pth', 
    map_location=device, weights_only=True,
)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, device

def pred_to_string(pred, remove_util=True):
    if remove_util: 
        pred = pred[pred != 11]

    return ','.join([str(i) for i  in pred.tolist()])

def generate_prediction(X, model, bos_token_id, eos_token_id):
    dec_input = torch.tensor(bos_token_id, dtype=torch.int64).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():   
        generated = dec_input  
        print(generated, generated.shape, generated.dtype)
        for idx in range(model.seq_len_dec - generated.size(1)):
            test_scores = model(X.unsqueeze(0)[0:1], generated)    # assume outputs.logits [1, T, V]
            # 3) Greedy pick at last position
            next_token = torch.argmax(test_scores[:, -1, :], dim=-1, keepdim=True)  # [1,1]
            st.write(f'Input: {pred_to_string(generated, False)}; Output: {int(next_token)}', )

            # 4) Append and check EOS
            generated = torch.cat([generated, next_token], dim=1)  # [1, T+1]
            if next_token.item() == eos_token_id:
                break
    return generated

# Load cached resources
train_ds, val_ds = load_mnist('~/data', True, patch_size=model_config['patch_size'])
model, device = prepare_model()

# Frontend -----------------------------------------------------
st.write("# 🔢 MNIST Grid captioner with transformers")

st.write("### Explore the validation dataset")

idx = st.slider(
    label="Select example index",
    min_value=0,
    max_value=len(val_ds) - 1,
    value=0,
    step=1
)
X, y = val_ds[idx]

st.image(
    unpatch(X, IM_SIZE, IM_SIZE, model_config['patch_size']).numpy(),
    caption='Ground truth Sequence: ' + pred_to_string(y),
    use_container_width=True
)
st.write("### 🛰️ Model prediction (autoregressive generation)")


generated = generate_prediction(X, model, val_ds.bos_token_id, val_ds.eos_token_id)
st.write('### 🪄 Final output: ' +pred_to_string(generated)) 

st.write('---')
st.write('## ✍️ Do it yourself!') 

upscale = 5

canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=15,
    stroke_color="rgba(255, 255, 255, 1)",
    background_color=[0,0,0],
    background_image=None,
    update_streamlit=True,
    height=IM_SIZE*upscale,
    width=IM_SIZE*upscale,
    drawing_mode='freedraw',
    point_display_radius=0,
    key="canvas",
)
X_input = canvas_result.image_data[::upscale,::upscale,0:1]

# st.image(
#     X_input,
#     # caption='Ground truth Sequence: ' + pred_to_string(y),
#     use_container_width=True
# )

dec_input = torch.tensor(val_ds.eos_token_id, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
enc_input = patchify(torch.tensor(X_input/255., dtype=torch.float32).permute(2,0,1), model_config['patch_size'])
st.write("### 🛰️ Model prediction (autoregressive generation)")

preds = generate_prediction(enc_input, model, val_ds.bos_token_id, val_ds.eos_token_id)
st.write('### 🪄 Final output: ' +pred_to_string(preds)) 
