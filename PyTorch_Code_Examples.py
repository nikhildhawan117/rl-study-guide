"""
PyTorch Fundamentals - Runnable Code Examples
==============================================

This file contains all examples from the study guide in runnable form.
Work through these interactively to build intuition.

Run with: python PyTorch_Code_Examples.py
Or run sections interactively in a Python REPL / Jupyter notebook.

Requirements: pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math

# Set seed for reproducibility
torch.manual_seed(42)


# ============================================================================
# SECTION 1: TENSOR BASICS
# ============================================================================


def tensor_basics():
    """Explore tensor creation and basic operations."""
    print("\n" + "=" * 60)
    print("SECTION 1: TENSOR BASICS")
    print("=" * 60)

    # Creation
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"\nCreated tensor:\n{x}")
    print(f"Shape: {x.shape}")
    print(f"Dtype: {x.dtype}")

    # Common initializations
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3)
    rand = torch.rand(2, 3)  # Uniform [0, 1)
    randn = torch.randn(2, 3)  # Normal(0, 1)

    print(f"\nZeros:\n{zeros}")
    print(f"\nRandom (uniform):\n{rand}")
    print(f"\nRandom (normal):\n{randn}")

    # Like another tensor
    like_x = torch.zeros_like(randn)
    print(f"\nzeros_like randn (same shape/device/dtype):\n{like_x}")

    # Dtype matters!
    float_tensor = torch.tensor([1.0, 2.0])
    int_tensor = torch.tensor([1, 2])
    print(f"\nFloat tensor dtype: {float_tensor.dtype}")  # float32
    print(f"Int tensor dtype: {int_tensor.dtype}")  # int64


def tensor_operations():
    """Common tensor operations you'll use constantly."""
    print("\n" + "=" * 60)
    print("TENSOR OPERATIONS")
    print("=" * 60)

    x = torch.randn(2, 3, 4)
    print(f"\nOriginal shape: {x.shape}")

    # Reshaping
    print(f"view(6, 4): {x.view(6, 4).shape}")
    print(f"view(-1, 4) (-1 infers): {x.view(-1, 4).shape}")
    print(f"flatten(): {x.flatten().shape}")
    print(f"flatten(start_dim=1): {x.flatten(start_dim=1).shape}")

    # Adding/removing dimensions
    y = torch.randn(3, 4)
    print(f"\nOriginal: {y.shape}")
    print(f"unsqueeze(0): {y.unsqueeze(0).shape}")  # Add dim at front
    print(f"unsqueeze(-1): {y.unsqueeze(-1).shape}")  # Add dim at end
    print(f"[None, :, :]: {y[None, :, :].shape}")  # Same as unsqueeze(0)

    z = torch.randn(1, 3, 1, 4)
    print(f"\nBefore squeeze: {z.shape}")
    print(f"squeeze(): {z.squeeze().shape}")  # Remove all size-1 dims

    # Transposing
    a = torch.randn(2, 3, 4)
    print(f"\nOriginal: {a.shape}")
    print(f"transpose(0, 1): {a.transpose(0, 1).shape}")
    print(f"permute(2, 0, 1): {a.permute(2, 0, 1).shape}")

    # Concatenation vs Stacking
    m = torch.randn(2, 3)
    n = torch.randn(2, 3)
    print(f"\nTwo tensors of shape: {m.shape}")
    print(f"cat dim=0: {torch.cat([m, n], dim=0).shape}")  # (4, 3)
    print(f"cat dim=1: {torch.cat([m, n], dim=1).shape}")  # (2, 6)
    print(f"stack dim=0: {torch.stack([m, n], dim=0).shape}")  # (2, 2, 3)


def autograd_demo():
    """Understanding automatic differentiation."""
    print("\n" + "=" * 60)
    print("SECTION 2: AUTOGRAD")
    print("=" * 60)

    # Basic gradient computation
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x * 2  # Each element doubled
    z = y.sum()  # Sum to scalar

    print(f"\nx = {x}")
    print(f"y = x * 2 = {y}")
    print(f"z = y.sum() = {z}")

    z.backward()  # Compute gradients
    print(f"dz/dx = {x.grad}")  # Should be [2, 2, 2]

    # Gradient accumulation (common gotcha!)
    print("\n--- Gradient Accumulation Demo ---")
    x = torch.tensor([1.0], requires_grad=True)

    # First backward
    (x * 2).sum().backward()
    print(f"After first backward: x.grad = {x.grad}")  # [2]

    # Second backward WITHOUT zeroing
    (x * 3).sum().backward()
    print(f"After second backward (accumulated!): x.grad = {x.grad}")  # [5] not [3]!

    # Proper way: zero gradients
    x.grad.zero_()
    (x * 3).sum().backward()
    print(f"After zeroing and backward: x.grad = {x.grad}")  # [3]

    # Detach demo
    print("\n--- Detach Demo ---")
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    y = x * 2
    y_detached = y.detach()  # Breaks gradient connection
    print(f"y.requires_grad: {y.requires_grad}")
    print(f"y_detached.requires_grad: {y_detached.requires_grad}")


# ============================================================================
# SECTION 2: nn.Module PATTERNS
# ============================================================================


class SimpleLinear(nn.Module):
    """The most basic custom module."""

    def __init__(self, in_features, out_features):
        super().__init__()  # MUST call this!
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLPFromScratch(nn.Module):
    """MLP with explicit parameter management (educational)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # Manual parameter creation
        self.W1 = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(hidden_dim))
        self.W2 = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b2 = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x):
        # x: (batch, input_dim)
        h = x @ self.W1 + self.b1  # (batch, hidden_dim)
        h = F.relu(h)
        out = h @ self.W2 + self.b2  # (batch, output_dim)
        return out


class MLPWithModules(nn.Module):
    """MLP using nn.Linear (the standard way)."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MLPSequential(nn.Module):
    """MLP using nn.Sequential (most concise)."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def module_demo():
    """Demonstrate nn.Module patterns."""
    print("\n" + "=" * 60)
    print("SECTION 3: nn.Module PATTERNS")
    print("=" * 60)

    # Create models
    model1 = MLPFromScratch(10, 32, 2)
    model2 = MLPWithModules(10, 32, 2)
    model3 = MLPSequential(10, 32, 2)

    # Test forward pass
    x = torch.randn(5, 10)  # Batch of 5

    print("\nForward pass outputs:")
    print(f"MLPFromScratch: {model1(x).shape}")
    print(f"MLPWithModules: {model2(x).shape}")
    print(f"MLPSequential: {model3(x).shape}")

    # Inspect parameters
    print("\n--- Parameter Inspection ---")
    print(f"MLPFromScratch parameters:")
    for name, param in model1.named_parameters():
        print(f"  {name}: {param.shape}")

    print(f"\nMLPWithModules parameters:")
    for name, param in model2.named_parameters():
        print(f"  {name}: {param.shape}")

    # Total parameter count
    total_params = sum(p.numel() for p in model2.parameters())
    print(f"\nTotal parameters: {total_params:,}")


# ============================================================================
# SECTION 3: TRAINING LOOP
# ============================================================================


def training_loop_demo():
    """The canonical PyTorch training loop."""
    print("\n" + "=" * 60)
    print("SECTION 4: TRAINING LOOP")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Synthetic classification data
    num_samples = 200
    input_dim = 20
    num_classes = 3

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # Create data loader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create model
    model = MLPSequential(input_dim, 64, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("\nTraining:")
    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()  # Training mode
        total_loss = 0.0

        for batch_inputs, batch_targets in train_loader:
            # Move to device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass
            logits = model(batch_inputs)
            loss = criterion(logits, batch_targets)

            # Backward pass
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # Evaluation
    model.eval()  # Evaluation mode
    with torch.no_grad():
        all_logits = model(X.to(device))
        predictions = all_logits.argmax(dim=-1)
        accuracy = (predictions.cpu() == y).float().mean()
        print(f"\nFinal accuracy: {accuracy:.4f}")


# ============================================================================
# SECTION 4: LINEAR REGRESSION EXAMPLE
# ============================================================================


def linear_regression_manual():
    """Linear regression with manual gradient descent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1A: LINEAR REGRESSION (Manual)")
    print("=" * 60)

    # Generate data: y = 2x + 1 + noise
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

    # Initialize parameters
    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    learning_rate = 0.1
    print(f"\nInitial: w={w.item():.4f}, b={b.item():.4f}")

    for epoch in range(100):
        # Forward
        y_pred = X * w + b
        loss = ((y_pred - y_true) ** 2).mean()

        # Backward
        loss.backward()

        # Update (manually)
        with torch.no_grad():
            w -= learning_rate * w.grad
            b -= learning_rate * b.grad

        # Zero gradients
        w.grad.zero_()
        b.grad.zero_()

        if (epoch + 1) % 25 == 0:
            print(
                f"Epoch {epoch+1}: Loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}"
            )

    print(f"\nLearned: w={w.item():.4f} (true: 2.0), b={b.item():.4f} (true: 1.0)")


def linear_regression_module():
    """Linear regression the proper PyTorch way."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1B: LINEAR REGRESSION (nn.Module)")
    print("=" * 60)

    # Generate data
    torch.manual_seed(42)
    X = torch.randn(100, 1)
    y_true = 2 * X + 1 + 0.1 * torch.randn(100, 1)

    # Define model
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(
        f"\nInitial: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}"
    )

    for epoch in range(100):
        y_pred = model(X)
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            w = model.linear.weight.item()
            b = model.linear.bias.item()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, w={w:.4f}, b={b:.4f}")

    print(
        f"\nLearned: w={model.linear.weight.item():.4f}, b={model.linear.bias.item():.4f}"
    )


# ============================================================================
# SECTION 5: MLP CLASSIFICATION EXAMPLE
# ============================================================================


def mlp_classification():
    """Full MLP classification example."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: MLP CLASSIFICATION")
    print("=" * 60)

    # Hyperparameters
    input_dim = 784  # MNIST-like
    hidden_dim = 256
    num_classes = 10
    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Synthetic data
    torch.manual_seed(42)
    num_samples = 1000
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # Split
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    # Model
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    print("\nTraining:")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()

        accuracy = correct / len(X_test)
        print(
            f"  Epoch {epoch+1}: Loss={train_loss/len(train_loader):.4f}, Test Acc={accuracy:.4f}"
        )


# ============================================================================
# SECTION 6: GPT-STYLE TRANSFORMER
# ============================================================================


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerMLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, embed_dim, expansion=4, dropout=0.1):
        super().__init__()
        hidden = embed_dim * expansion
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block."""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = TransformerMLP(embed_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Minimal GPT."""

    def __init__(
        self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.token_emb.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape

        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.pos_emb.num_embeddings :]
            logits = self(idx_cond)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def gpt_example():
    """Train a tiny GPT."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: GPT-STYLE TRANSFORMER")
    print("=" * 60)

    # Tiny config
    vocab_size = 100
    embed_dim = 64
    num_heads = 4
    num_layers = 2
    max_seq_len = 32
    batch_size = 16
    num_steps = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPT(vocab_size, embed_dim, num_heads, num_layers, max_seq_len).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Synthetic data
    torch.manual_seed(42)
    data = torch.randint(0, vocab_size, (200, max_seq_len + 1))
    inputs = data[:, :-1]
    targets = data[:, 1:]

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    print("\nTraining:")
    model.train()
    for step in range(num_steps):
        # Sample batch
        idx = torch.randint(0, len(inputs), (batch_size,))
        x = inputs[idx].to(device)
        y = targets[idx].to(device)

        # Forward
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}: Loss={loss.item():.4f}")

    # Generate
    print("\nGeneration:")
    model.eval()
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated tokens: {generated[0].tolist()}")


# ============================================================================
# SECTION 7: USEFUL PATTERNS FOR RL
# ============================================================================


def rl_patterns():
    """Common patterns you'll use in RL post-training."""
    print("\n" + "=" * 60)
    print("BONUS: PATTERNS FOR RL POST-TRAINING")
    print("=" * 60)

    # Setup a simple "model"
    vocab_size = 100
    model = nn.Embedding(vocab_size, 32)
    proj = nn.Linear(32, vocab_size)

    def get_logits(token_ids):
        return proj(model(token_ids))

    # Example: Get log probabilities of specific tokens
    print("\n--- Getting Log Probabilities ---")
    input_ids = torch.randint(0, vocab_size, (4, 10))  # (batch=4, seq=10)
    target_ids = torch.randint(0, vocab_size, (4, 10))  # tokens we want probs of

    logits = get_logits(input_ids)  # (4, 10, 100)
    log_probs_all = F.log_softmax(logits, dim=-1)  # (4, 10, 100)

    # Gather log probs of target tokens
    log_probs = log_probs_all.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(
        -1
    )  # (4, 10)

    print(f"Input shape: {input_ids.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Sample log probs: {log_probs[0, :5]}")

    # Example: REINFORCE-style gradient
    print("\n--- REINFORCE Gradient Pattern ---")
    rewards = torch.tensor([1.0, -0.5, 0.8, 0.2])  # Reward per sequence
    baseline = rewards.mean()
    advantages = rewards - baseline

    # Policy gradient: -log_prob * advantage
    sequence_log_probs = log_probs.sum(dim=-1)  # (4,) - sum over tokens
    policy_loss = -(sequence_log_probs * advantages).mean()

    print(f"Rewards: {rewards}")
    print(f"Baseline: {baseline:.3f}")
    print(f"Advantages: {advantages}")
    print(f"Policy loss: {policy_loss.item():.4f}")

    # Example: KL divergence between policies
    print("\n--- KL Divergence ---")
    # Simulate log probs from policy and reference
    log_probs_policy = torch.randn(4, 10)
    log_probs_ref = torch.randn(4, 10)

    # Approximate KL (commonly used)
    approx_kl = (log_probs_ref - log_probs_policy).mean()
    print(f"Approximate KL: {approx_kl.item():.4f}")

    # Example: Detaching for target networks
    print("\n--- Detaching (Stop Gradient) ---")
    x = torch.randn(3, requires_grad=True)
    target = (x * 2).detach()  # No gradient flows through target
    loss = ((x - target) ** 2).mean()
    loss.backward()
    print(f"x.grad: {x.grad}")  # Gradient only from x, not from target


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("PYTORCH FUNDAMENTALS - CODE EXAMPLES")
    print("=" * 60)

    # Section 1: Basics
    tensor_basics()
    tensor_operations()
    autograd_demo()

    # Section 2: Modules
    module_demo()

    # Section 3: Training
    training_loop_demo()

    # Section 4: Linear Regression
    linear_regression_manual()
    linear_regression_module()

    # Section 5: MLP Classification
    mlp_classification()

    # Section 6: Transformer
    gpt_example()

    # Section 7: RL Patterns
    rl_patterns()

    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
