# PyTorch Fundamentals Study Guide

**Context:** For ML practitioners who understand the theory but need hands-on PyTorch implementation skills. Assumes familiarity with backprop, gradient descent, neural networks, and transformers conceptually.

---

## Table of Contents

1. [Tensors: The Foundation](#1-tensors-the-foundation)
   - [Creating Tensors](#11-creating-tensors)
   - [Tensor Operations](#12-tensor-operations)
   - [NumPy Interop](#13-numpy-interop)
   - [Device Management (CPU/GPU)](#14-device-management-cpugpu)
2. [Autograd: Automatic Differentiation](#2-autograd-automatic-differentiation)
   - [The Big Picture: What Are We Computing and Why?](#20-the-big-picture-what-are-we-computing-and-why)
   - [requires_grad and the Computational Graph](#21-requires_grad-and-the-computational-graph)
   - [backward() and Gradient Accumulation](#22-backward-and-gradient-accumulation)
   - [Detaching and no_grad](#23-detaching-and-no_grad)
3. [nn.Module: The Core Abstraction](#3-nnmodule-the-core-abstraction)
   - [Anatomy of a Module](#31-anatomy-of-a-module)
   - [Parameters vs Buffers](#32-parameters-vs-buffers)
   - [Nested Modules](#33-nested-modules)
   - [Common Built-in Layers (with explanations)](#34-common-built-in-layers)
4. [The Training Loop](#4-the-training-loop)
   - [Optimizer Mechanics](#41-optimizer-mechanics)
   - [The Canonical Loop](#42-the-canonical-loop)
   - [train() vs eval() Mode](#43-train-vs-eval-mode)
   - [Saving and Loading](#44-saving-and-loading)
5. [Loss Functions](#5-loss-functions)
   - [Common Losses](#51-common-losses)
   - [Reduction Modes](#52-reduction-modes)
6. [Data Loading](#6-data-loading)
   - [Dataset and DataLoader](#61-dataset-and-dataloader)
   - [Batching and Shuffling](#62-batching-and-shuffling)
7. [Example 1: Linear Regression from Scratch](#7-example-1-linear-regression-from-scratch)
8. [Example 2: MLP for Classification](#8-example-2-mlp-for-classification)
9. [Example 3: GPT-Style Transformer](#9-example-3-gpt-style-transformer)
10. [Common Patterns and Idioms](#10-common-patterns-and-idioms)
11. [Debugging Tips](#11-debugging-tips)
12. [Quick Reference](#12-quick-reference)

---

## 1. Tensors: The Foundation

### 1.1 Creating Tensors

Tensors are PyTorch's fundamental data structure—multi-dimensional arrays with automatic differentiation support.

```python
import torch

# From Python lists
x = torch.tensor([1, 2, 3])                    # Shape: (3,)
x = torch.tensor([[1, 2], [3, 4]])             # Shape: (2, 2)

# Specifying dtype (default is float32 for floats, int64 for ints)
x = torch.tensor([1.0, 2.0], dtype=torch.float32)
x = torch.tensor([1, 2], dtype=torch.long)     # int64, needed for indices

# Common initialization patterns
zeros = torch.zeros(3, 4)                       # Shape: (3, 4)
ones = torch.ones(3, 4)
rand = torch.rand(3, 4)                         # Uniform [0, 1)
randn = torch.randn(3, 4)                       # Normal(0, 1)
arange = torch.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
linspace = torch.linspace(0, 1, 5)              # [0, 0.25, 0.5, 0.75, 1]

# Like another tensor (same shape, dtype, device)
y = torch.zeros_like(x)
y = torch.randn_like(x)

# Identity matrix
eye = torch.eye(3)                              # 3x3 identity
```

**Key dtype reminder:** Most operations expect `float32`. Indices (for embeddings, gather, etc.) must be `long` (int64).

### 1.2 Tensor Operations

```python
# Shape manipulation
x = torch.randn(2, 3, 4)
x.shape                          # torch.Size([2, 3, 4])
x.size()                         # Same as .shape
x.numel()                        # Total elements: 24

# Reshaping (VERY common)
x.view(6, 4)                     # Reshape to (6, 4) - must be contiguous
x.reshape(6, 4)                  # Like view but handles non-contiguous
x.view(-1, 4)                    # -1 infers dimension: (6, 4)
x.flatten()                      # (24,) - 1D
x.flatten(start_dim=1)           # (2, 12) - flatten dims 1 and 2

# Adding/removing dimensions
x = torch.randn(3, 4)
x.unsqueeze(0)                   # (1, 3, 4) - add dim at position 0
x.unsqueeze(-1)                  # (3, 4, 1) - add dim at end
x[None, :, :]                    # Same as unsqueeze(0)
x[:, :, None]                    # Same as unsqueeze(-1)

y = torch.randn(1, 3, 4)
y.squeeze()                      # (3, 4) - remove all dims of size 1
y.squeeze(0)                     # (3, 4) - remove dim 0 only if size 1

# Transposing
x = torch.randn(2, 3, 4)
x.T                              # Only for 2D: transpose
x.transpose(0, 1)                # Swap dims 0 and 1: (3, 2, 4)
x.permute(2, 0, 1)              # Arbitrary reorder: (4, 2, 3)

# Concatenation and stacking
a = torch.randn(2, 3)
b = torch.randn(2, 3)
torch.cat([a, b], dim=0)         # (4, 3) - concat along dim 0
torch.cat([a, b], dim=1)         # (2, 6) - concat along dim 1
torch.stack([a, b], dim=0)       # (2, 2, 3) - NEW dim at position 0

# Indexing (same as NumPy)
x = torch.randn(3, 4, 5)
x[0]                             # First element of dim 0: (4, 5)
x[:, 1]                          # Second element of dim 1: (3, 5)
x[..., -1]                       # Last element of last dim: (3, 4)
x[:, 1:3]                        # Slice: (3, 2, 5)

# Boolean indexing
mask = x > 0
x[mask]                          # 1D tensor of positive values

# Advanced indexing with gather (used in getting specific token logits)
indices = torch.tensor([[0, 1], [2, 0]])  # Shape: (2, 2)
src = torch.randn(3, 4)
# gather picks elements: out[i][j] = src[i][indices[i][j]]
out = src.gather(dim=1, index=indices)    # Shape: (2, 2)
```

### 1.3 NumPy Interop

```python
import numpy as np

# NumPy → Tensor (SHARES memory by default!)
np_array = np.array([1.0, 2.0, 3.0])
tensor = torch.from_numpy(np_array)        # Shares memory
tensor = torch.tensor(np_array)            # Copies data

# Tensor → NumPy (must be on CPU)
tensor = torch.randn(3)
np_array = tensor.numpy()                  # Shares memory
np_array = tensor.detach().cpu().numpy()   # Safe pattern for GPU tensors
```

### 1.4 Device Management (CPU/GPU)

```python
# Check availability
torch.cuda.is_available()                  # True if CUDA GPU available
torch.cuda.device_count()                  # Number of GPUs

# Device specification
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')            # Specific GPU
device = torch.device('cpu')

# Moving tensors
x = torch.randn(3, 4)
x = x.to(device)                           # Move to device
x = x.cuda()                               # Shorthand for GPU
x = x.cpu()                                # Shorthand for CPU

# Creating directly on device
x = torch.randn(3, 4, device=device)

# Moving models (more on this later)
model = model.to(device)
```

**Critical rule:** Tensors must be on the same device to operate together. You'll get errors like `Expected all tensors to be on the same device`.

---

## 2. Autograd: Automatic Differentiation

### 2.0 The Big Picture: What Are We Computing and Why?

Before diving into syntax, let's understand what autograd is actually doing.

**The goal:** You have some computation that ends in a scalar (a loss). You want to know: _"If I nudge each input value a little, how much does the loss change?"_ That's the gradient—it tells you how to adjust your weights to reduce the loss.

```python
# Step 1: Create a tensor and tell PyTorch "I want gradients for this"
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Step 2: Do some operations (PyTorch secretly builds a computation graph)
y = x * 2        # y = [2, 4, 6]
z = y.sum()      # z = 12 (must end in a scalar to call backward)

# Step 3: Compute gradients: "how does z change if I change x?"
z.backward()     # Computes dz/dx

print(x.grad)    # tensor([2., 2., 2.])
```

**What just happened?** Let's do the math:

```
z = sum(y) = sum(x * 2) = 2*x[0] + 2*x[1] + 2*x[2]

dz/dx[0] = 2
dz/dx[1] = 2
dz/dx[2] = 2

So x.grad = [2, 2, 2]
```

**The gradient tells you:** "If I increase `x[0]` by a tiny amount ε, `z` increases by 2ε."

**Why this matters for training:**

```python
# In a neural network, this is what happens:
loss = compute_loss(model(input), target)  # scalar loss
loss.backward()  # computes d(loss)/d(every weight in model)

# Now each weight has a .grad attribute telling you:
# "If you increase this weight, does the loss go up or down, and by how much?"

# So you update weights in the OPPOSITE direction of the gradient:
weight = weight - learning_rate * weight.grad  # gradient descent!
```

**The computation graph PyTorch builds:**

```
x (leaf tensor, requires_grad=True)
│
▼  multiply by 2
y (intermediate, has grad_fn=MulBackward)
│
▼  sum
z (output scalar, has grad_fn=SumBackward)
│
▼  z.backward()
│
Gradients flow BACKWARDS through the graph (chain rule)
x.grad gets populated with dz/dx
```

**Key insight:** Gradients are always computed **of a scalar** (the loss) **with respect to tensors** (the weights/parameters you want to optimize).

### 2.1 requires_grad and the Computational Graph

Now let's look at the mechanics in more detail.

```python
# By default, tensors don't track gradients
x = torch.tensor([1.0, 2.0, 3.0])
x.requires_grad                             # False

# Enable gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# OR
x.requires_grad_(True)                      # In-place toggle

# Operations on requires_grad tensors create a graph
y = x * 2
z = y.sum()

# Each tensor knows how it was created (for backward pass)
z.grad_fn                                   # <SumBackward0>
y.grad_fn                                   # <MulBackward0>
x.grad_fn                                   # None (leaf tensor - you created it)
```

**Leaf tensors:** Tensors you create directly (not computed from other tensors) are "leaves." Only leaf tensors store `.grad` after `backward()`.

### 2.2 backward() and Gradient Accumulation

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()                                 # Must be scalar to call backward()

# Compute gradients
z.backward()                                # Populates x.grad

print(x.grad)                               # tensor([2., 2., 2.])
                                            # dz/dx = 2 for each element

# CRITICAL: Gradients ACCUMULATE by default!
z = (x * 3).sum()
z.backward()
print(x.grad)                               # tensor([5., 5., 5.]) NOT [3, 3, 3]!
                                            # It added 3 to the previous 2

# You must zero gradients before each backward pass
x.grad.zero_()                              # In-place zero
# Or for optimizers:
optimizer.zero_grad()                       # Zeros all param grads
```

**Why accumulate?** Useful for gradient accumulation across mini-batches (simulating larger batch sizes when GPU memory is limited).

### 2.3 Detaching and no_grad

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2

# detach() creates a tensor that shares data but no grad connection
y_detached = y.detach()
y_detached.requires_grad                    # False

# Common use: treating a value as a constant
# Example: target in RL shouldn't propagate gradients
target = compute_target(state).detach()

# no_grad context: temporarily disable gradient tracking
with torch.no_grad():
    # Operations here don't build graph (faster, less memory)
    y = model(x)
    y.requires_grad                         # False

# Common use: inference/evaluation
model.eval()
with torch.no_grad():
    predictions = model(test_data)
```

**When to use each:**

- `detach()`: Stop gradients from flowing through a specific value
- `no_grad()`: Disable gradient computation entirely (inference)
- `inference_mode()`: Even stricter than no_grad, slightly faster (PyTorch 1.9+)

---

## 3. nn.Module: The Core Abstraction

### 3.1 Anatomy of a Module

Every neural network component in PyTorch inherits from `nn.Module`. Here's the pattern:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()  # MUST call this first!

        # Define layers/submodules as attributes
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

        # You can also define raw parameters
        self.my_param = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x):
        """Define the forward pass. This is what gets called when you do model(x)."""
        x = self.layer1(x)
        x = self.activation(x)
        x = x + self.my_param  # Use the parameter
        x = self.layer2(x)
        return x

# Usage
model = MyModel(input_dim=10, hidden_dim=32, output_dim=2)
x = torch.randn(5, 10)  # Batch of 5, input dim 10
output = model(x)       # Calls forward() internally. Shape: (5, 2)
```

**The two methods you must know:**

1. `__init__`: Define all layers, parameters, and submodules as attributes
2. `forward`: Define computation. Called via `model(x)`, not `model.forward(x)`

**Why `super().__init__()`?** It initializes the internal machinery that tracks parameters, enables `.to(device)`, `.train()/.eval()`, etc.

### 3.2 Parameters vs Buffers

```python
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

        # Parameters: learned weights, updated by optimizer
        self.weight = nn.Parameter(torch.randn(10, 10))

        # Buffers: saved state that's NOT learned (e.g., running stats)
        self.register_buffer('running_mean', torch.zeros(10))

        # Regular attributes: not saved, not moved to device automatically
        self.some_constant = 3.14

# Access parameters
model = MyModule()
list(model.parameters())        # [weight tensor]
list(model.named_parameters())  # [('weight', tensor)]

# Buffers are saved in state_dict but not in parameters()
model.state_dict()              # {'weight': ..., 'running_mean': ...}

# Both params and buffers move with .to(device)
model.to('cuda')
model.running_mean.device       # cuda:0
```

**Common buffer uses:**

- BatchNorm running mean/variance
- Positional encoding tables
- Attention masks that don't change

### 3.3 Nested Modules

```python
class EncoderBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.mlp(self.norm(x))

class Encoder(nn.Module):
    def __init__(self, dim, num_layers):
        super().__init__()

        # ModuleList: for layers you'll iterate over
        self.layers = nn.ModuleList([
            EncoderBlock(dim) for _ in range(num_layers)
        ])

        # Sequential: for simple feed-forward stacks
        self.output_head = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

# All nested parameters are tracked automatically
model = Encoder(dim=64, num_layers=3)
len(list(model.parameters()))   # Counts params in all submodules
```

**ModuleList vs Python list:**

```python
# WRONG - parameters not registered!
self.layers = [nn.Linear(10, 10) for _ in range(3)]

# RIGHT - parameters registered properly
self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])
```

**ModuleDict** for named submodules:

```python
self.heads = nn.ModuleDict({
    'classification': nn.Linear(dim, num_classes),
    'regression': nn.Linear(dim, 1)
})
```

**nn.Sequential — Chaining layers without writing forward()**

`nn.Sequential` is a container that runs layers in order. You give it a list of modules, and it automatically feeds each layer's output as the next layer's input.

```python
# Without Sequential: you write forward() explicitly
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# With Sequential: forward() is automatic
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)  # Just call it!
```

**What Sequential does internally:**

```python
# When you call self.net(x), Sequential does this:
def forward(self, x):
    for module in self.children():
        x = module(x)
    return x
```

**When to use Sequential:**

- Simple linear chains where each layer's output feeds directly into the next
- MLPs, simple CNNs, any "stack of layers" pattern

**When NOT to use Sequential:**

- Skip connections / residual connections (output needs to be added to input)
- Multiple inputs or outputs
- Conditional logic in forward pass
- Anything where you need to do more than just chain operations

```python
# Can't use Sequential for residual connections:
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.layers(x)  # Residual: need to add x back!
```

**Accessing layers inside Sequential:**

```python
seq = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

seq[0]              # First layer: Linear(10, 20)
seq[1]              # Second layer: ReLU()
seq[-1]             # Last layer: Linear(20, 5)

# Named version for clearer access:
seq = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(10, 20)),
    ('relu', nn.ReLU()),
    ('fc2', nn.Linear(20, 5))
]))

seq.fc1             # Access by name
```

**Note: Dimension mismatches are runtime errors, not compile-time:**

```python
# This creates without error — PyTorch doesn't check dimensions until you run it
broken = nn.Sequential(nn.Linear(10, 20), nn.Linear(50, 5))  # 20 ≠ 50!
broken(torch.randn(32, 10))  # RuntimeError only here
```

**Note: Most activations are element-wise (dimension doesn't matter):**

```python
# ReLU, GELU, Tanh, Sigmoid — apply independently to every element
F.relu(x)      # max(0, x) for each element, any shape works

# Softmax is the exception — needs a dim because it normalizes (values sum to 1)
F.softmax(x, dim=-1)  # Normalizes along last dimension
```

### 3.4 Common Built-in Layers

#### nn.Linear — Fully Connected / Dense Layer

The workhorse of neural networks. Applies a linear transformation: `output = input @ weight.T + bias`

```python
linear = nn.Linear(in_features=768, out_features=256)
x = torch.randn(32, 768)       # (batch, in_features)
out = linear(x)                 # (32, 256)
```

Each output is a weighted sum of all inputs. Has `weight` of shape `(out, in)` and `bias` of shape `(out,)`.

#### nn.Embedding — Lookup Table

Converts integer indices into dense vectors. This is how tokens become embeddings in transformers.

```python
embed = nn.Embedding(num_embeddings=50000, embedding_dim=768)  # 50k vocab, 768-dim
token_ids = torch.tensor([[1, 42, 1000], [7, 8, 9]])           # (batch=2, seq_len=3)
vectors = embed(token_ids)                                      # (2, 3, 768)
```

Internally it's just a matrix of shape `(vocab_size, embed_dim)`. The forward pass is a table lookup, not a matrix multiply.

#### nn.LayerNorm — Layer Normalization

Normalizes activations across the feature dimension (independently per sample). Used extensively in transformers.

```python
ln = nn.LayerNorm(normalized_shape=768)
x = torch.randn(32, 100, 768)  # (batch, seq_len, features)
out = ln(x)                     # Same shape, but normalized
```

For each position in each sample: subtracts mean, divides by std, then applies learnable scale (γ) and shift (β). Stabilizes training by keeping activations in a reasonable range.

#### nn.BatchNorm1d/2d — Batch Normalization

Normalizes across the batch dimension (statistics computed over all samples). Common in CNNs, less so in transformers.

```python
bn = nn.BatchNorm1d(num_features=256)
x = torch.randn(32, 256)       # (batch, features)
out = bn(x)                     # Same shape
```

Key difference from LayerNorm: computes mean/std _across the batch_, keeps running statistics for inference.

#### nn.Dropout — Regularization

Randomly zeros elements during training with probability `p`. Prevents overfitting.

```python
dropout = nn.Dropout(p=0.1)
x = torch.randn(32, 768)
out = dropout(x)               # ~10% of values are 0 (during training)
```

**Important:** Only active during `model.train()`. During `model.eval()`, dropout does nothing.

#### Activations — Non-linearities

Without activations, stacking linear layers would just be one big linear transformation.

```python
# ReLU: max(0, x) — simple, but "dead neurons" can be a problem
nn.ReLU()

# GELU: smooth approximation, used in BERT/GPT
# x * Φ(x) where Φ is the CDF of standard normal
nn.GELU()

# SiLU/Swish: x * sigmoid(x) — used in many modern architectures
nn.SiLU()

# Softmax: converts logits to probabilities (sums to 1)
nn.Softmax(dim=-1)
```

#### nn.Conv2d — Convolutional Layer (for images)

Slides a small kernel over the input, computing dot products. Captures local patterns.

```python
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
img = torch.randn(8, 3, 224, 224)  # (batch, channels, height, width)
out = conv(img)                     # (8, 64, 224, 224)
```

#### nn.LSTM / nn.GRU — Recurrent Layers

Process sequences step-by-step, maintaining hidden state. Largely replaced by transformers for NLP.

```python
lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, batch_first=True)
x = torch.randn(32, 100, 256)           # (batch, seq_len, features)
output, (h_n, c_n) = lstm(x)            # output: (32, 100, 512)
```

#### Functional vs Module versions

```python
import torch.nn.functional as F

# Module version (has state, use in __init__)
self.relu = nn.ReLU()
out = self.relu(x)

# Functional version (stateless, use directly in forward)
out = F.relu(x)
out = F.gelu(x)
out = F.softmax(x, dim=-1)
out = F.cross_entropy(logits, targets)  # Combines log_softmax + nll_loss
```

Use **modules** when you need learnable parameters or stateful behavior (Linear, LayerNorm, Dropout).
Use **functional** for pure operations with no state (activations, loss functions).

---

## 4. The Training Loop

### 4.1 Optimizer Mechanics

```python
import torch.optim as optim

# Create optimizer with model parameters
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# The three sacred steps:
optimizer.zero_grad()   # 1. Clear old gradients
loss.backward()         # 2. Compute new gradients
optimizer.step()        # 3. Update parameters

# What step() does under the hood (for SGD):
# param = param - lr * param.grad
```

**Different learning rates for different parts:**

```python
optimizer = optim.Adam([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(), 'lr': 1e-3}
])
```

**Learning rate scheduling:**

```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# In training loop:
for epoch in range(num_epochs):
    train_one_epoch()
    scheduler.step()  # Update LR after each epoch
```

### 4.2 The Canonical Loop

```python
def train(model, train_loader, optimizer, criterion, device, num_epochs):
    model.train()  # Set to training mode

    for epoch in range(num_epochs):
        total_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            loss.backward()        # Compute gradients

            # Optional: gradient clipping (common in transformers)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()       # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
```

### 4.3 train() vs eval() Mode

```python
model.train()   # Training mode
model.eval()    # Evaluation mode

# What changes:
# - Dropout: active in train(), disabled in eval()
# - BatchNorm: updates running stats in train(), uses them in eval()
# - Some custom modules may behave differently

# Evaluation loop
def evaluate(model, test_loader, criterion, device):
    model.eval()  # Set to eval mode
    total_loss = 0.0
    correct = 0

    with torch.no_grad():  # Disable gradient computation
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # For classification
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()

    return total_loss / len(test_loader), correct / len(test_loader.dataset)
```

### 4.4 Saving and Loading

```python
# Save just the weights (recommended)
torch.save(model.state_dict(), 'model_weights.pt')

# Load weights
model = MyModel(...)  # Create model with same architecture
model.load_state_dict(torch.load('model_weights.pt'))

# Save entire model (includes architecture, less portable)
torch.save(model, 'model_full.pt')
model = torch.load('model_full.pt')

# Save checkpoint (for resuming training)
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

---

## 5. Loss Functions

### 5.1 Common Losses

```python
import torch.nn as nn
import torch.nn.functional as F

# Mean Squared Error (regression)
criterion = nn.MSELoss()
loss = criterion(predictions, targets)  # Both shape (batch, *)

# Cross Entropy (classification) - COMBINES log_softmax + NLLLoss
criterion = nn.CrossEntropyLoss()
# predictions: (batch, num_classes) - RAW LOGITS, not softmax!
# targets: (batch,) - class indices as LONG tensor
loss = criterion(predictions, targets)

# Binary Cross Entropy
criterion = nn.BCEWithLogitsLoss()  # Takes raw logits
# predictions: (batch,) - raw logits
# targets: (batch,) - 0 or 1 as floats
loss = criterion(predictions, targets)

# Functional versions (useful for custom logic)
loss = F.cross_entropy(logits, targets)
loss = F.mse_loss(predictions, targets)
loss = F.binary_cross_entropy_with_logits(logits, targets)
```

**CrossEntropyLoss detail (important for LLM training):**

```python
# For language modeling, you have:
# logits: (batch, seq_len, vocab_size)
# targets: (batch, seq_len)

# Reshape for cross_entropy which expects (N, C) and (N,)
logits = logits.view(-1, vocab_size)      # (batch * seq_len, vocab_size)
targets = targets.view(-1)                 # (batch * seq_len,)
loss = F.cross_entropy(logits, targets)

# Or use ignore_index to skip padding tokens
loss = F.cross_entropy(logits, targets, ignore_index=PAD_TOKEN_ID)
```

### 5.2 Reduction Modes

```python
# Default: mean over all elements
criterion = nn.MSELoss(reduction='mean')

# Sum instead of mean
criterion = nn.MSELoss(reduction='sum')

# No reduction - get loss per element
criterion = nn.MSELoss(reduction='none')
per_element_loss = criterion(pred, target)  # Shape matches input
# Useful for weighting losses differently
```

---

## 6. Data Loading

### 6.1 Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a single sample
        return self.data[idx], self.labels[idx]

# Create dataset
dataset = MyDataset(X_train, y_train)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,         # Shuffle each epoch
    num_workers=4,        # Parallel data loading
    pin_memory=True,      # Faster GPU transfer
    drop_last=True        # Drop incomplete final batch
)

# Iterate
for batch_inputs, batch_targets in train_loader:
    # batch_inputs: (32, ...) - batched automatically
    pass
```

### 6.2 Batching and Shuffling

```python
# For variable-length sequences, use collate_fn
def collate_fn(batch):
    """Custom batching logic."""
    inputs, targets = zip(*batch)

    # Pad sequences to same length
    inputs_padded = nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets_padded = nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return inputs_padded, targets_padded

loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
```

---

## 7. Example 1: Linear Regression from Scratch

Let's implement linear regression to see the core PyTorch mechanics.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generate synthetic data: y = 2x + 1 + noise
torch.manual_seed(42)
X = torch.randn(100, 1)
y = 2 * X + 1 + 0.1 * torch.randn(100, 1)

# Method 1: Using raw tensors and autograd
w = torch.randn(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

learning_rate = 0.1
for epoch in range(100):
    # Forward pass
    y_pred = X * w + b

    # Compute loss
    loss = ((y_pred - y) ** 2).mean()  # MSE

    # Backward pass
    loss.backward()

    # Update weights (manually, no optimizer)
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    # Zero gradients for next iteration
    w.grad.zero_()
    b.grad.zero_()

print(f"Learned: w={w.item():.4f}, b={b.item():.4f}")
# Expected: w≈2.0, b≈1.0
```

```python
# Method 2: Using nn.Module (the standard way)
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    # Forward
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Access learned parameters
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"Learned: w={w:.4f}, b={b:.4f}")
```

**Key takeaways:**

- `nn.Module` handles parameter registration automatically
- The optimizer pattern (`zero_grad`, `backward`, `step`) is universal
- `nn.Linear` includes both weight and bias by default

---

## 8. Example 2: MLP for Classification

A multi-layer perceptron for MNIST-style classification.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
            # Note: no softmax! CrossEntropyLoss expects raw logits
        )

    def forward(self, x):
        return self.net(x)

# Create synthetic classification data
torch.manual_seed(42)
num_samples, input_dim, num_classes = 1000, 784, 10
X = torch.randn(num_samples, input_dim)
y = torch.randint(0, num_classes, (num_samples,))  # Random labels

# Split into train/test
train_dataset = TensorDataset(X[:800], y[:800])
test_dataset = TensorDataset(X[800:], y[800:])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP(input_dim=784, hidden_dim=256, num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward
        logits = model(inputs)
        loss = criterion(logits, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()

    accuracy = correct / len(test_dataset)
    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.4f}")
```

**Key takeaways:**

- `nn.Sequential` chains layers without needing explicit forward
- CrossEntropyLoss expects raw logits, not softmax output
- `.train()` and `.eval()` toggle dropout behavior
- Classification accuracy: compare `argmax` of logits to targets

---

## 9. Example 3: GPT-Style Transformer

A minimal GPT implementation. This is causal (autoregressive) language modeling.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embed dim

        # Project to Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Causal mask: prevent attending to future tokens
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # (B, num_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)

        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward network in transformer block."""

    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden_dim = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block: attention + MLP with residuals and layer norm."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=dropout)

    def forward(self, x):
        # Pre-norm architecture (like GPT-2)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Minimal GPT-style language model."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers,
                 max_seq_len, dropout=0.1):
        super().__init__()

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output head
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying (common practice)
        self.token_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        Args:
            idx: (batch, seq_len) tensor of token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape

        # Get embeddings
        tok_emb = self.token_emb(idx)  # (B, T, embed_dim)
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.pos_emb(pos)    # (T, embed_dim)

        x = self.dropout(tok_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop to max_seq_len if needed
            idx_cond = idx[:, -self.pos_emb.num_embeddings:]

            # Get predictions
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # Last position only

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# Example usage
def train_gpt_example():
    # Hyperparameters (tiny for demo)
    vocab_size = 1000
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    max_seq_len = 32
    batch_size = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = GPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Synthetic data: random token sequences
    data = torch.randint(0, vocab_size, (100, max_seq_len + 1))

    # For language modeling: input is tokens[:-1], target is tokens[1:]
    inputs = data[:, :-1].to(device)
    targets = data[:, 1:].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Training loop
    model.train()
    for step in range(100):
        # Sample batch
        idx = torch.randint(0, len(inputs), (batch_size,))
        x, y = inputs[idx], targets[idx]

        # Forward
        logits = model(x)

        # Compute loss (reshape for cross_entropy)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # Generate
    model.eval()
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated tokens: {generated[0].tolist()}")


if __name__ == "__main__":
    train_gpt_example()
```

**Key GPT implementation details:**

1. **Causal masking:** `torch.triu` creates upper triangular mask; we fill with -inf so softmax gives 0 attention to future tokens

2. **Weight tying:** `self.token_emb.weight = self.lm_head.weight` shares parameters between embedding and output projection (common practice, reduces parameters)

3. **Pre-norm vs Post-norm:** We use pre-norm (LayerNorm before attention/MLP), which is more stable for training

4. **Position embeddings:** Learned absolute positions; alternatives include rotary (RoPE) or sinusoidal

5. **Language modeling loss:** Input is `tokens[:-1]`, target is `tokens[1:]` (predict next token)

---

## 10. Common Patterns and Idioms

### Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)

model.apply(init_weights)  # Recursively applies to all submodules
```

### Gradient Clipping

```python
# By norm (most common)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# By value
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

### Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss = loss / accumulation_steps  # Scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in train_loader:
    optimizer.zero_grad()

    with autocast():  # Auto-cast to fp16 where safe
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Freezing Parameters

```python
# Freeze entire model
for param in model.parameters():
    param.requires_grad = False

# Freeze specific layers
for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze head for fine-tuning
for param in model.head.parameters():
    param.requires_grad = True
```

### Getting Log Probabilities (for RL)

```python
def get_log_probs(model, input_ids, target_ids):
    """
    Get log probabilities of target tokens under the model.
    Used in REINFORCE, PPO, DPO, etc.
    """
    logits = model(input_ids)  # (batch, seq_len, vocab_size)

    # Log softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

    # Gather log probs of actual tokens
    # target_ids: (batch, seq_len)
    log_probs = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)  # (batch, seq_len)

    return log_probs
```

### KL Divergence Between Models

```python
def kl_divergence(log_probs_p, log_probs_q):
    """
    KL(P || Q) = sum P(x) * (log P(x) - log Q(x))

    For RL: measures how far policy has drifted from reference.
    """
    return (log_probs_p.exp() * (log_probs_p - log_probs_q)).sum(dim=-1)


def approximate_kl(log_probs_policy, log_probs_ref):
    """
    Simpler approximation often used in practice:
    KL ≈ (log_probs_ref - log_probs_policy).mean()
    """
    return (log_probs_ref - log_probs_policy).mean()
```

---

## 11. Debugging Tips

### Shape Debugging

```python
# Print shapes at each step
def forward(self, x):
    print(f"Input: {x.shape}")
    x = self.layer1(x)
    print(f"After layer1: {x.shape}")
    # ... etc
```

### Gradient Debugging

```python
# Check if gradients are flowing
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad mean={param.grad.mean():.6f}, std={param.grad.std():.6f}")
    else:
        print(f"{name}: no gradient!")

# Check for NaN/Inf
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                print(f"Inf gradient in {name}")
```

### Common Errors

| Error                                                                    | Likely Cause                                                |
| ------------------------------------------------------------------------ | ----------------------------------------------------------- |
| `RuntimeError: CUDA out of memory`                                       | Batch too large, or not freeing intermediate tensors        |
| `Expected all tensors on same device`                                    | Mixing CPU and GPU tensors                                  |
| `shape mismatch`                                                         | Wrong dimensions in matrix multiply or concat               |
| `one of the variables needed for gradient computation has been modified` | In-place operation on tensor needed for backward            |
| `grad can be implicitly created only for scalar outputs`                 | Called `.backward()` on non-scalar without passing gradient |

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Delete tensors and clear
del large_tensor
torch.cuda.empty_cache()
```

---

## 12. Quick Reference

### Essential Imports

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
```

### Module Template

```python
class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define layers

    def forward(self, x):
        # Define computation
        return output
```

### Training Template

```python
model.train()
for inputs, targets in loader:
    inputs, targets = inputs.to(device), targets.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Evaluation Template

```python
model.eval()
with torch.no_grad():
    for inputs, targets in loader:
        outputs = model(inputs.to(device))
        # compute metrics
```

### Key Shape Operations

| Operation                    | Input            | Output                     |
| ---------------------------- | ---------------- | -------------------------- |
| `x.view(a, b)`               | `(m, n)`         | `(a, b)` where `a*b = m*n` |
| `x.unsqueeze(0)`             | `(a, b)`         | `(1, a, b)`                |
| `x.squeeze(0)`               | `(1, a, b)`      | `(a, b)`                   |
| `x.transpose(0, 1)`          | `(a, b)`         | `(b, a)`                   |
| `x.permute(2, 0, 1)`         | `(a, b, c)`      | `(c, a, b)`                |
| `torch.cat([x, y], dim=0)`   | `(a, b), (a, b)` | `(2a, b)`                  |
| `torch.stack([x, y], dim=0)` | `(a, b), (a, b)` | `(2, a, b)`                |

### Loss Function Inputs

| Loss                | Predictions                     | Targets                 |
| ------------------- | ------------------------------- | ----------------------- |
| `MSELoss`           | `(batch, *)` float              | `(batch, *)` float      |
| `CrossEntropyLoss`  | `(batch, classes)` float logits | `(batch,)` long indices |
| `BCEWithLogitsLoss` | `(batch,)` float logits         | `(batch,)` float 0/1    |
| `NLLLoss`           | `(batch, classes)` log probs    | `(batch,)` long indices |

---

## Further Resources

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Andrej Karpathy's minGPT](https://github.com/karpathy/minGPT) - Clean GPT implementation
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - Even simpler GPT
- [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/) - Line-by-line transformer explanation
