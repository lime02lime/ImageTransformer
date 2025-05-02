# Transformer

Task is to make an image captioning network.

We compose an image using MNIST tokens and then want to caption them.

## Dataset

MNIST Dataset

- 28 x 28 hand-drawn numbers

## Tasks

### Task 1: Use transformer encoder to extract an embedding for classification

Models: CNN
Loss function: Cross entropy loss

| Model                               | Number of parameters | Training time/ epoch [s] | Validation Accuracy (5 epochs) [%] |
| ----------------------------------- | -------------------- | ------------------------ | ---------------------------------- |
| CNN                                 | 61706                | 10                       | 98.6                               |
| Transformer 3 layers, 1 head        | 5092                 | 70                       | 92.0                               |
| Transformer 3 layers, 1 head (fast) | 5092                 | 30                       |                                    |
| Transformer, 12 layers, 1 head      | 18700                | 100                      | 91.4                               |
| Transformer, 12 layers, 3 heads     |                      |                          |                                    |

The transformer has been implemented with learnable positional encodings.

### Task 2: OCR image captioning

| Model                                                                              | Number of parameters |
| ---------------------------------------------------------------------------------- | -------------------- |
| ViT Encoder (16x16 patch size, 128 model dim, pre-norm, GELU), Transformer decoder | 836748               |

![Example image](media/patched_example.png)

The task is to use a vision transformer, trained from scratch to encode the image, and then use a transformer to decode it.

A nice baseline would be CNN-based character detector and then the MNIST classifier.

    Comparisons:
    Similarities
        -
    Differences:
        - ViT
            - is just an encoder
            - Implements
        -
    Implements Pre normalisation

First I tried grid predictions

- Remember to pad the tokens

Learnings

- Names are hard
- Start from the specific, then generalise. Write out the loopy version, write out the matmul version
- Think about inputs and outputs and what they mean
- Make sure you test your inputs and outputs, do sanity checks
- Data leakage
  - Softmax issue
  - Shifting tokens
- Reduce your problem to something manageable

Transformer decoder:

- The job of the language modeling head is to take the output of the final trans-
  former layer from the last token N and use it to predict the upcoming word at posi-
  tion N + 1
- Decoder inference is the hard bit

```bash
# First epoch
# First guesses the sequence length, and then nothing in the loss matters
INFO:__main__:Correct seq:      2,3,7,2,3,11,10,10,10,10,10,10
INFO:__main__:Predicted seq:    2,3,3,3,3,11,11,11,11,11,11,11
```

```python
# To visualise the weights
import matplotlib.pyplot as plt
_,N_heads,N_tokens, _ = A.shape
for i in range(N_heads):
    plt.matshow(A[0][i].cpu().detach().numpy())
    plt.ylabel('Q')
    plt.xlabel('K')
    plt.title('Masked self-attention for decoder')
    plt.show()
    break

import matplotlib.pyplot as plt
_,N_heads,N_tokens, _ = A.shape
for i in range(N_tokens):
    plt.matshow(A[0][1][i].cpu().detach().numpy().reshape(7,7))
    plt.title(f'Cross attention visualisation for dec token {i}')
    plt.show()
```

Todo

- Implement masking in the predictor
- Try CNN + RNN Encoder/Decoder setup
- Data augmentations all the way

## Training setup

First time on device

```bash
source setup_env.sh
```

Next

```
conda activate venv
```

## References
