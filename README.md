# ML Study Guides for Post-Training Roles

This repository contains comprehensive study guides for ML engineering interviews, particularly for LLM post-training positions.

---

## Contents

### RL for LLM Post-Training

Covers reinforcement learning algorithms used in LLM post-training, particularly useful for code post-training and reasoning applications.

### PyTorch Fundamentals

Covers PyTorch implementation from the ground up—tensors, autograd, nn.Module patterns, and practical examples including a GPT implementation.

---

## Setup

To run the code examples, set up a Python environment with PyTorch:

```bash
# Clone the repo
git clone <repo-url>
cd rl-study-guide

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run PyTorch examples
python PyTorch_Code_Examples.py

# Run RL examples
python Code_Examples.py
```

**Requirements:**

- Python 3.9+
- PyTorch 2.0+ (tested with 2.8.0)
- NumPy

---

## Study Guide Contents

### 1. [RL_Post_Training_Study_Guide.md](./RL_Post_Training_Study_Guide.md) (Main Document)

The comprehensive main guide covering:

- **Foundational Concepts**

  - MDP formulation for LLMs
  - Key symbols and definitions (π, V, Q, A, r, etc.)
  - Policy gradient theorem
  - Value functions vs reward models
  - Token-level vs trajectory-level actions

- **Algorithm Deep Dives**

  - REINFORCE and variants (RLOO, REINFORCE++)
  - PPO (Proximal Policy Optimization)
  - A2C (Advantage Actor-Critic)
  - GRPO (Group Relative Policy Optimization)
  - DPO (Direct Preference Optimization)
  - IPO (Identity Preference Optimization)

- **Advanced Topics**

  - Distributed training architectures
  - Bradley-Terry reward model training
  - KL divergence penalty
  - On-policy vs off-policy training
  - Process vs outcome reward models

- **Resources**
  - Links to key papers
  - Implementation resources
  - Interview preparation checklist

### 2. [Quick_Reference_Sheet.md](./Quick_Reference_Sheet.md)

One-page reference for quick review:

- Core equations at a glance
- Algorithm decision tree
- Hyperparameter cheat sheet
- Common interview questions
- Recent trends (2024-2025)

### 3. [Deep_Dive_QA.md](./Deep_Dive_QA.md)

Extended answers to commonly confused concepts:

- What exactly is Q? Ground truth or learned?
- How is the value function trained during RL?
- Rollouts and advantage computation timing
- Code execution feedback as a special case
- Mental model for LLM RL

### 4. [Code_Examples.py](./Code_Examples.py)

Pedagogical implementations of key algorithms:

- Bradley-Terry loss
- REINFORCE with baseline
- RLOO
- PPO with GAE
- GRPO
- DPO
- KL divergence computation
- Training loop examples

### 5. [PyTorch_Fundamentals_Study_Guide.md](./PyTorch_Fundamentals_Study_Guide.md) (NEW)

Comprehensive PyTorch guide for practitioners who know ML theory but need implementation skills:

- **Core Concepts**

  - Tensors (creation, operations, dtype, reshaping)
  - Autograd (computational graph, backward, gradient accumulation)
  - Device management (CPU/GPU)

- **nn.Module Deep Dive**

  - Anatomy of a module (`__init__`, `forward`)
  - Parameters vs buffers
  - Nested modules (ModuleList, Sequential)
  - Common built-in layers

- **Training Mechanics**

  - Optimizer patterns (zero_grad, backward, step)
  - The canonical training loop
  - train() vs eval() mode
  - Saving and loading checkpoints

- **Practical Examples**

  - Linear regression (manual and nn.Module)
  - MLP for classification
  - GPT-style transformer (full implementation)

- **Patterns for RL**
  - Getting log probabilities
  - KL divergence computation
  - Gradient clipping, freezing parameters
  - Mixed precision training

### 6. [PyTorch_Code_Examples.py](./PyTorch_Code_Examples.py) (NEW)

Runnable code for all PyTorch examples:

- Tensor operations playground
- Autograd demonstrations
- Module patterns comparison
- Complete training loops
- Linear regression (manual + module)
- MLP classification
- Minimal GPT implementation
- RL-specific patterns (log probs, KL, detach)

---

## Suggested Study Order

### For RL Post-Training (Original)

### Week 1: Foundations

1. Read Sections 1-3 of main guide (MDP, Policy Gradients, REINFORCE)
2. Work through the REINFORCE code examples
3. Read the "What is Q?" section in Deep Dive

### Week 2: Core Algorithms

1. Read Sections 4-6 of main guide (PPO, A2C, GRPO)
2. Study the value function training section in Deep Dive
3. Work through PPO and GRPO code examples

### Week 3: Preference Learning

1. Read Sections 7-8 of main guide (DPO, IPO)
2. Understand Bradley-Terry model thoroughly
3. Work through DPO code examples

### Week 4: Advanced Topics & Review

1. Read Sections 9-10 of main guide (Distributed, Advanced)
2. Review Quick Reference Sheet daily
3. Practice explaining each algorithm from memory
4. Go through the interview preparation checklist

### For PyTorch Fundamentals (2-3 hours)

**Hour 1: Core Concepts**

1. Read Sections 1-2 (Tensors, Autograd)
2. Run `tensor_basics()`, `tensor_operations()`, `autograd_demo()` in the code examples
3. Key goal: Understand `requires_grad`, `backward()`, gradient accumulation

**Hour 2: nn.Module and Training**

1. Read Sections 3-6 (Module, Training Loop, Loss, Data)
2. Run `module_demo()`, `training_loop_demo()`
3. Run `linear_regression_manual()` and `linear_regression_module()` to see both approaches
4. Key goal: Be able to write a training loop from memory

**Hour 3: Advanced Examples**

1. Read Sections 7-10 (Examples, Patterns)
2. Run `mlp_classification()` and `gpt_example()`
3. Study the GPT code carefully—understand each component
4. Run `rl_patterns()` to see how this connects to post-training
5. Key goal: Understand transformer implementation and RL-specific patterns

---

## Key Papers to Read

### Essential (Read Fully)

1. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. **DPO**: Rafailov et al. "Direct Preference Optimization" (2023)
3. **DeepSeek-R1**: GRPO and rule-based rewards for reasoning

### Recommended (Skim + Key Sections)

4. **GAE**: Schulman et al. "High-Dimensional Continuous Control Using GAE" (2015)
5. **IPO**: Azar et al. "A General Theoretical Paradigm..." (2023)
6. **RLOO**: Ahmadian et al. "Back to Basics..." (2024)

### Optional (Awareness)

7. **LlamaRL**: Distributed RL architecture
8. **REINFORCE++**: Global advantage normalization
9. **KL-Regularised Q-Learning**: Token-level action values

---

## Interview Topics by Importance

### Must Know (Will Definitely Be Asked)

- [ ] Policy gradient theorem derivation
- [ ] PPO clipping mechanism
- [ ] DPO loss derivation from RLHF objective
- [ ] Value function vs reward model distinction
- [ ] KL divergence penalty and why it's needed

### Should Know (Likely to Come Up)

- [ ] GAE and the bias-variance tradeoff
- [ ] Why GRPO doesn't need a critic
- [ ] Bradley-Terry model for reward training
- [ ] Token-level vs trajectory-level RL
- [ ] On-policy vs off-policy considerations

### Good to Know (May Impress)

- [ ] IPO vs DPO theoretical differences
- [ ] RLOO's unbiased baseline
- [ ] Distributed RL architectures (samplers, learners, etc.)
- [ ] Process reward models
- [ ] Recent trends in critic-free methods

---

## Quick Self-Test Questions

Before an interview (or for self-testing), readers should be able to answer:

1. **Derive the policy gradient theorem** from first principles using the log-derivative trick.

2. **Explain why subtracting a baseline** from the reward doesn't introduce bias.

3. **Walk through PPO's clipping mechanism** - what happens when the ratio exceeds 1.2 for a positive advantage?

4. **Write the DPO loss** and explain why it's equivalent to RLHF.

5. **Explain how GRPO computes advantages** without a value function.

6. **How is KL divergence computed** in practice? Monte Carlo or full distribution?

7. **What's the difference between** the reward model, value function, and advantage?

8. **When would one use GRPO vs PPO** vs DPO?

---

## Notes for Code Post-Training Context

For readers working on or interested in code post-training:

- **Verifiable rewards** are a major advantage—code either passes tests or doesn't
- **GRPO** is particularly relevant; DeepSeek-R1 showed it works well with binary rewards
- **Process reward models** may be worth discussing for multi-step code generation
- **Execution feedback** eliminates the need for learned reward models
- **Credit assignment** is less of an issue when step-by-step verification is available

---

## Contact & Updates

This guide was prepared February 2026. Check for updates to algorithms and best practices, as the field moves quickly.

Good luck with your studies and interview preparation!
