# Vision Transformer (ViT) Implementation

This repository contains an implementation of a Vision Transformer (ViT) using PyTorch. The code is designed to train and evaluate the model on the MNIST dataset.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Testing](#testing)
- [Acknowledgements](#acknowledgements)

## Installation To run this code, you'll need to have Python installed along with the required libraries. You can install the necessary dependencies using pip:
```bash
pip install torch torchvision numpy tqdm
```
# Model Architecture
** Vision Transformer (ViT) **
The ViT class defines the Vision Transformer model:

# Parameters:

* chw (tuple): Channels, height, and width of the input images.
* n_patches (int): Number of patches to divide the image into.
* n_heads (int): Number of attention heads in the multi-head self-attention.
* n_blocks (int): Number of encoder blocks.
* hidden_d (int): Hidden dimension size.
* out_d (int): Output dimension size (number of classes).

# Components:

* Linear Mapper: A linear layer to map the flattened patches to the hidden dimension.
* Class Token: A learnable token for classification.
* Positional Embeddings: Positional encodings to retain positional information.
* Encoder Blocks: A list of encoder blocks, each containing self-attention and feed-forward layers.
* MLP Head: A multi-layer perceptron for the final classification.
* Multi-Head Attention
* The MultiHeadAttention class implements the multi-head self-attention mechanism:

# Parameters:

* d (int): Dimension of the input.
* n_heads (int): Number of attention heads.

# Components:

* Linear layers for query, key, and value mappings for each attention head.
* Softmax layer for attention weights.

# Encoder Block
* The EncoderVIT class defines the encoder block used in ViT:

# Parameters:

* hidden_d (int): Hidden dimension size.
* n_heads (int): Number of attention heads.
* mlp_ratio (int): Expansion ratio for the MLP.
  
# Components:

* Layer normalization.
* Multi-head self-attention.
Feed-forward network with GELU activation.
* Training
* The main() function handles the training process:

# Data Preparation:

* Loads the MNIST dataset and converts images to tensors.
* Creates data loaders for training and testing.
* Device Setup: Chooses GPU if available, otherwise CPU.
* Model Initialization: Initializes the Vision Transformer model.
# Training Loop:
* Iterates over epochs and batches, performing forward pass, loss computation, backpropagation, and optimizer step.
* Logs training loss after each epoch.

# Testing
*  Evaluation: Evaluates the model on the test set without gradient computation.
* Accuracy Calculation: Computes accuracy based on model predictions.
* Logging: Prints test loss and accuracy.
  
# Example Output

Here is an example of the output you might see during training and testing:
```

Training:   0%|          | 0/5 [00:00<?, ?it/s]

Epoch 1 in training:   0%|          | 0/469 [00:00<?, ?it/s]

...
Epoch 1 loss: 0.0298
...
Epoch 5 loss: 0.0183

testing: 100%|██████████| 79/79 [00:00<00:00, 92.01it/s]

Test loss: 0.0180

Accuracy: 0.9932
```
# Acknowledgements
This implementation is inspired by the Vision Transformer paper and adapted for educational purposes to demonstrate the basic concepts.
