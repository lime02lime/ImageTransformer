import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoder_decoder import CustomTransformerEncoderDecoder, create_image_patches

# Load the trained model
@st.cache_resource
def load_model():
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../best_model_768D.pth'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'learning_rate': 1e-4,
        'min_lr': 5e-7,
        'weight_decay': 0.01,
        'epochs': 25,
        'batch_size': 128,
        'num_heads': 8,
        'emb_dim': 1024,
        'ff_hidden_dim': 128,
        'num_encoder_layers': 8,
        'num_decoder_layers': 8,
        'num_classes': 13,  # 10 digits + start/stop tokens
        'max_seq_length': 12,
        'patch_dim': 196,  # 14x14 pixel patches
        'num_patches': 64   # each image is 4x4 sub images each with 2x2 patches of 14x14 pixels
    }
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
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Streamlit app
st.title("Digit Sequence Prediction")
st.write("Draw digits on the canvas below and submit to get predictions.")

# Create two columns: left for canvas, right for predictions
col1, col2 = st.columns([2, 1])  # Adjust column width ratio as needed

with col1:
    # Canvas for user input
    canvas_result = st_canvas(
        fill_color="white",  # Background color
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        width=448,  # Larger canvas for easier drawing
        height=448,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.write("### Prediction Results")
    submit = st.button("Submit")

if submit:
    if canvas_result.image_data is not None:
        # Convert canvas image (numpy RGBA) to proper PIL image
        rgba = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        rgb = rgba.convert('RGB')
        gray = rgb.convert('L')  # Convert to grayscale

        # Resize to 112x112 using NEAREST to preserve stroke shape
        resized = gray.resize((112, 112), resample=Image.NEAREST)

        # Display the resized image in the left column
        #with col1:
        #    st.write("### Resized Image (112x112):")
        #    st.image(resized, caption="Resized Image (Input to Model)", width=224)

        # Convert to array and normalize
        img_array = np.array(resized) / 255.0  # Values in [0, 1]

        # Convert to PyTorch tensor and reshape
        img_tensor = torch.tensor(img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 112, 112]

        # Create patches
        patches = create_image_patches(img_tensor, patch_size=14)
        patches = torch.tensor(patches, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare input for the decoder
        tgt_input = torch.tensor([[10]], dtype=torch.long).to(device)  # START token

        # Predict sequence
        with torch.no_grad():
            for _ in range(12):  # Maximum sequence length
                output = model(patches, tgt_input)
                next_token = output[:, -1, :].argmax(dim=-1)
                tgt_input = torch.cat([tgt_input, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == 11:  # STOP token
                    break

        # Decode the predicted sequence
        predicted_sequence = tgt_input.squeeze().tolist()[1:]  # Remove START token
        readable_sequence = [str(token) if token < 10 else "STOP" for token in predicted_sequence]

        # Display the result in the right column
        with col2:
            st.write("### Predicted Sequence:")
            st.write(" → ".join(readable_sequence))
    else:
        st.warning("Please draw something on the canvas before submitting.")