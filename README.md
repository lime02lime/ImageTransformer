# ImageTransformer: Grid-Based MNIST Sequence Prediction

This project implements a transformer-based model to predict sequences of digits from grid-based MNIST images. The pipeline includes dataset preparation, model training, evaluation, and hyperparameter sweeps using Weights & Biases (WandB).

## Features
- **Custom Dataset**: Combines MNIST digits into grid-based images with start, stop, and padding tokens for sequence prediction.
- **Transformer Model**: A custom encoder-decoder transformer architecture for sequence generation.
- **Training & Evaluation**: Includes training loops, evaluation metrics, and visualization of predictions.
- **Hyperparameter Sweeps**: Integrated with WandB for automated hyperparameter optimization.

## Model Overview
The core of this project is a custom transformer-based encoder-decoder model. The model processes grid-based MNIST images and predicts sequences of digits in an autoregressive manner. This means that the model generates one token at a time, using previously generated tokens as input for the next prediction. The sequence starts with a special start token (10) and ends with a stop token (11). Padding tokens (12) are used to ensure consistent sequence lengths during training.

### Key Features of the Model
- **Encoder**: Processes image patches and extracts meaningful representations.
- **Decoder**: Generates sequences token by token, conditioned on the encoder's output and previously generated tokens.
- **Autoregressive Generation**: Ensures that each token prediction depends on the context of prior tokens, enabling accurate sequence generation.

## Project Structure
- `data_prep.py`: Prepares the grid-based MNIST dataset and handles data loading.
- `model.py`: Defines the custom transformer encoder-decoder model.
- `main.py`: Main script for training and evaluating the model.
- `main_sweep.py`: Script for running hyperparameter sweeps with WandB.
- `requirements.txt`: Python dependencies for the project.

## Setup
1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the MNIST dataset (handled automatically during dataset creation).

## Usage
### Training
Run the `main.py` script to train the model:
```bash
python encoder_decoder/main.py
```
You will be prompted to choose between training or evaluation.

### Evaluation
To evaluate the model, ensure a trained model checkpoint exists and select "eval" when prompted.

### Hyperparameter Sweeps
Run `main_sweep.py` to perform hyperparameter optimization:
```bash
python encoder_decoder/main_sweep.py
```
Configure sweep parameters in `sweep.yaml` or directly in the script.

## Running Hyperparameter Sweeps
To initialize hyperparameter sweeps using Weights & Biases (WandB), follow these steps:

1. Ensure you have configured your sweep parameters in `sweep.yaml`.
2. Run the following command in the terminal to start the sweep (this will return a sweep ID):
   ```bash
   wandb sweep encoder_decoder/sweep.yaml
   ```
3. Once the sweep is initialized, launch agents to execute the runs:
   ```bash
   wandb agent <SWEEP_ID>
   ```
   Replace `<SWEEP_ID>` with the ID provided by WandB after initializing the sweep.

## Outputs
- **Model Checkpoints**: Saved in the project directory (e.g., `best_model.pth`).
- **Visualization**: Example predictions are saved as images (e.g., `example_0.png`).

## Notes
- Start token: `10`
- Stop token: `11`
- Padding token: `12`

## Dependencies
- Python 3.8+
- PyTorch
- WandB
- Matplotlib
- NumPy
- TorchVision

## Acknowledgments
This project uses the MNIST dataset.