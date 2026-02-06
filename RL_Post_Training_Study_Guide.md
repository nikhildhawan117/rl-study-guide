# Comprehensive Reinforcement Learning Study Guide for LLM Post-Training

**Context:** This guide is especially relevant for code and reasoning post-training applications, but covers RL for LLM post-training broadly.

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
   - [Markov Decision Process (MDP) Formulation for LLMs](#11-markov-decision-process-mdp-formulation-for-llms)
   - [Key Symbols and Definitions](#12-key-symbols-and-definitions)
   - [Policy Gradients: The Foundation](#13-policy-gradients-the-foundation)
   - [Value Functions vs Reward Models](#14-value-functions-vs-reward-models)
   - [Token-Level vs Trajectory-Level Actions](#15-token-level-vs-trajectory-level-actions)
2. [Reward Model Training](#2-reward-model-training)
   - [Bradley-Terry Model](#21-bradley-terry-model)
   - [Practical Considerations](#22-practical-considerations)
3. [REINFORCE Algorithm](#3-reinforce-algorithm)
   - [Core Algorithm](#31-core-algorithm)
   - [Variance Reduction Techniques](#32-variance-reduction-techniques)
   - [REINFORCE++ and RLOO](#33-reinforce-and-rloo)
4. [Proximal Policy Optimization (PPO)](#4-proximal-policy-optimization-ppo)
   - [Trust Region Methods](#41-trust-region-methods)
   - [PPO Clipped Objective](#42-ppo-clipped-objective)
   - [Generalized Advantage Estimation (GAE)](#43-generalized-advantage-estimation-gae)
   - [Actor-Critic Architecture](#44-actor-critic-architecture)
5. [A2C: Advantage Actor-Critic](#5-a2c-advantage-actor-critic)
6. [Group Relative Policy Optimization (GRPO)](#6-group-relative-policy-optimization-grpo)
   - [Core Innovation](#61-core-innovation)
   - [Mathematical Formulation](#62-mathematical-formulation)
   - [KL Divergence Penalty](#63-kl-divergence-penalty)
   - [DeepSeek-R1 Implementation Details](#64-deepseek-r1-implementation-details)
7. [Direct Preference Optimization (DPO)](#7-direct-preference-optimization-dpo)
   - [Theoretical Foundation](#71-theoretical-foundation)
   - [DPO Loss Derivation](#72-dpo-loss-derivation)
   - [Implicit Reward Model](#73-implicit-reward-model)
8. [Identity Preference Optimization (IPO)](#8-identity-preference-optimization-ipo)
   - [DeepMind's Approach](#81-deepminds-approach)
   - [Comparison with DPO](#82-comparison-with-dpo)
9. [Distributed Training in RL](#9-distributed-training-in-rl)
   - [Architecture Paradigms](#91-architecture-paradigms)
   - [Key Components: Samplers, Trainers, Learners, Actors, Critics](#92-key-components-samplers-trainers-learners-actors-critics)
10. [Advanced Topics](#10-advanced-topics)
    - [On-Policy vs Off-Policy Training](#101-on-policy-vs-off-policy-training)
    - [Importance Sampling](#102-importance-sampling)
    - [Process Reward Models vs Outcome Reward Models](#103-process-reward-models-vs-outcome-reward-models)
11. [Algorithm Comparison Summary](#11-algorithm-comparison-summary)
12. [Further Reading and Resources](#12-further-reading-and-resources)

---

## 1. Foundational Concepts

### 1.1 Markov Decision Process (MDP) Formulation for LLMs

In LLM post-training, we formulate text generation as a Markov Decision Process:

| MDP Component                         | LLM Interpretation                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------- |
| **State** $s_t$                       | Prompt $x$ concatenated with tokens generated so far: $s_t = (x, y_1, y_2, \ldots, y_{t-1})$ |
| **Action** $a_t$                      | Next token $y_t$ from vocabulary $\mathcal{V}$                                               |
| **Policy** $\pi_\theta(a_t \mid s_t)$ | LLM's probability distribution over next token given context                                 |
| **Reward** $r_t$                      | Typically 0 for intermediate tokens; final reward at sequence end                            |
| **Trajectory** $\tau$                 | Complete response $(y_1, y_2, \ldots, y_T)$                                                  |
| **Episode**                           | One complete generation from prompt to end token                                             |

**Critical Distinction**: There are two common formulations:

1. **Token-Level MDP**: Each token is an action, requiring $Q$ and $V$ estimates at every token position
2. **Trajectory-Level MDP**: The entire response is one "action," reward is given only at the end

Most modern LLM RL uses a **hybrid approach**: token-level policy but trajectory-level (or turn-level) rewards.

### 1.2 Key Symbols and Definitions

| Symbol              | Definition                                                                                       | Notes                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------- |
| $\pi_\theta$        | Policy (the LLM being trained) parameterized by $\theta$                                         |                                                                                                         |
| $\pi_{\text{ref}}$  | Reference policy (frozen base model)                                                             | Used for KL regularization                                                                              |
| $r(x, y)$           | Reward for response $y$ given prompt $x$                                                         | From reward model or rule-based                                                                         |
| $R(y)$ or $R(\tau)$ | Return (cumulative reward) for trajectory                                                        | Often just terminal reward in LLM setting                                                               |
| $V^{\pi}(s)$        | **State Value Function**: Expected return starting from state $s$ following policy $\pi$.        | $V^{\pi}(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t \text{ given } s_0 = s\right]$            |
| $Q^{\pi}(s, a)$     | **Action-Value Function**: Expected return taking action $a$ in state $s$, then following $\pi$. | $Q^{\pi}(s,a) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t \text{ given } s_0 = s, a_0 = a\right]$ |
| $A^\pi(s, a)$       | **Advantage Function**: How much better action $a$ is compared to average                        | $A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$                                                                  |
| $\gamma$            | Discount factor                                                                                  | Often 1.0 in LLM setting (finite horizon)                                                               |
| $\beta$             | KL penalty coefficient                                                                           | Controls deviation from reference                                                                       |
| $\epsilon$          | PPO clipping parameter                                                                           | Typically 0.2                                                                                           |

### 1.3 Policy Gradients: The Foundation

#### SFT: A Quick Recap

Here is what happens in SFT/pretraining:

```
SFT Training Step:
─────────────────
Prompt: "What is 2+2?"
Target: "The answer is 4."

For EACH position, model sees everything BEFORE that position and predicts the next token:

Position 1: sees "What is 2+2?"           → predict "The"    ✓
Position 2: sees "What is 2+2? The"       → predict "answer" ✓
Position 3: sees "What is 2+2? The answer"→ predict "is"     ✓
Position 4: sees "What is 2+2? The answer is" → predict "4"  ✓

Loss = sum of cross-entropy at ALL positions
     = -log P("The") - log P("answer") - log P("is") - log P("4")

We maximize probability of the correct next token at EVERY position.
```

**SFT Loss:**
$$L_{\text{SFT}} = -\sum_{t=1}^{T} \log \pi_\theta(y_t^* | x, y_{<t}^*)$$

In English: "Sum up the negative log probability of each correct token. Minimize this, which means maximize the probability of the correct tokens."

**Why is there no "difference" term?** You might expect cross-entropy to compare two distributions explicitly. Here's why it simplifies:

$$L_{\text{cross-entropy}} = -\sum_{v \in \text{vocab}} p_{\text{target}}(v) \cdot \log p_{\text{model}}(v)$$

When target is **one-hot** (probability 1 for correct token, 0 for everything else):

$$L = -\underbrace{1}_{\text{correct}} \cdot \log p_{\text{model}}(\text{"4"}) - \underbrace{0}_{\text{wrong}} \cdot \log p_{\text{model}}(\text{"5"}) - \underbrace{0}_{\text{wrong}} \cdot \log p_{\text{model}}(\text{"the"}) - \ldots$$

All the zeros kill the other terms, leaving:

$$L = -\log p_{\text{model}}(\text{"4"})$$

So cross-entropy with one-hot targets **is** just the negative log-prob of the correct token. The "difference" is implicit: minimizing $-\log p(\text{correct})$ automatically pushes probability toward the correct token (since probabilities must sum to 1).

**Key property of SFT**: The correct token is known at every position.

---

#### Now, How Does RL Differ from SFT?

**The fundamental problem**: In RL, we DON'T have a "correct" token at each position. We only have:

- A complete response the model generated
- A reward score for that complete response (e.g., "this response scored 0.7")

**The question becomes**: How do we update the model when we don't have per-token ground truth?

**The RL answer**: Instead of saying "this token is correct, increase its probability," we say "this response got a good reward, so increase the probability of ALL the tokens in it."

---

#### SFT vs RL: Side-by-Side Comparison

| Aspect                 | SFT                                          | RL (Policy Gradient)                       |
| ---------------------- | -------------------------------------------- | ------------------------------------------ |
| **Signal**             | Per-token ground truth                       | Per-response reward score                  |
| **What we optimize**   | "Make P(correct token) high"                 | "Make P(high-reward responses) high"       |
| **Gradient direction** | Push toward the one correct token            | Push toward tokens that led to good reward |
| **Weighting**          | All tokens weighted equally (or by position) | Tokens weighted by how good the reward was |

---

#### The Policy Gradient: What We Actually Optimize

**Goal**: Maximize expected reward

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[r(x, y)]$$

**In English**: "On average, across all prompts and the responses our model generates, we want the reward to be high."

**The Policy Gradient Theorem** gives us:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t\right]$$

**In English, piece by piece**:

| Symbol                                      | Plain English                                                                     |
| ------------------------------------------- | --------------------------------------------------------------------------------- |
| $\nabla_\theta J(\theta)$                   | "The direction we should move the model weights to increase expected reward"      |
| $\mathbb{E}_{\tau \sim \pi_\theta}$         | "Average over responses sampled from our current model"                           |
| $\sum_{t=0}^{T}$                            | "Sum over all tokens in the response"                                             |
| $\nabla_\theta \log \pi_\theta(a_t \| s_t)$ | "The direction that increases probability of token $a_t$" (same gradient as SFT!) |
| $R_t$                                       | "How good was this response?" (the reward, or advantage)                          |

**The key insight**: The gradient $\nabla_\theta \log \pi_\theta(a_t | s_t)$ is IDENTICAL to what is computed in SFT. The difference is:

- **SFT**: Weight = 1 for correct token, 0 for everything else
- **RL**: Weight = Reward (or advantage) for the token that was actually sampled

---

#### Concrete Example: SFT vs RL Update

**Setup**: Model generated response "The answer is 5" to "What is 2+2?"

**SFT** (if we had ground truth "The answer is 4"):

```
Token "5" at final position:
- Ground truth says it should be "4"
- Gradient: DECREASE P("5"), INCREASE P("4")
- This happens regardless of anything else
```

**RL** (we got reward = -1 for this wrong response):

```
Token "5" at final position:
- We sampled "5", and the whole response got reward -1
- Gradient: DECREASE P("5") (weighted by -1)
- "5" was what we picked, and it led to bad reward, so make it less likely

Token "answer" earlier in response:
- We sampled "answer", and the whole response got reward -1
- Gradient: DECREASE P("answer") slightly
- Even correct intermediate tokens get blamed a little (credit assignment problem!)
```

**RL** (if we had gotten reward = +1):

```
Token "5" at final position:
- We sampled "5", and the whole response got reward +1
- Gradient: INCREASE P("5") (weighted by +1)
- Even though "5" is wrong, if the reward model liked it, we reinforce it!
```

---

#### Why Advantage Instead of Raw Reward?

If we use raw reward $R$, we have problems:

```
Response A: reward = 8.5  → Increase probability of all tokens (by 8.5)
Response B: reward = 8.2  → Increase probability of all tokens (by 8.2)
Response C: reward = 8.0  → Increase probability of all tokens (by 8.0)
```

There _is_ a relative signal—higher rewards get reinforced more. But (1) the magnitude is tiny (8.5 vs 8.0), and (2) we _never_ decrease below-average responses—we only increase them less. With all positive rewards, every response gets its probability boosted; we're not explicitly discouraging the worse ones.

**Solution: Use advantage** $A = R - \text{baseline}$

```
Average reward = 8.23

Response A: advantage = 8.5 - 8.23 = +0.27  → Increase probability (slightly)
Response B: advantage = 8.2 - 8.23 = -0.03  → Decrease probability (slightly)
Response C: advantage = 8.0 - 8.23 = -0.23  → Decrease probability
```

Now we're asking: "Was this response BETTER or WORSE than average?" This gives a meaningful learning signal.

---

#### The Complete Picture: RL Loss for LLMs

Here's what we actually minimize:

$$L_{\text{RL}}(\theta) = -\mathbb{E}_{x, y \sim \pi_\theta}\left[\big(\sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})\big) \cdot A(x, y)\right]$$

**In plain English**:

1. Sample a response $y$ from the model for prompt $x$
2. Compute the sum of log-probs for each token (just like SFT!)
3. Multiply by the advantage (how much better than average was this response?)
4. If advantage > 0: Push UP the probability of all these tokens
5. If advantage < 0: Push DOWN the probability of all these tokens

**Compare to SFT**:

$$L_{\text{SFT}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(y_t^* | x, y_{<t}^*)$$

The only differences are:

1. RL uses sampled tokens $y$, SFT uses ground truth tokens $y^*$
2. RL multiplies by advantage $A$, SFT implicitly uses weight 1

---

#### Summary: The Mental Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    SFT vs RL Mental Model                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SFT:    "Here's the correct answer. Learn to produce it."      │
│          Loss = -log P(correct tokens)                           │
│          Gradient = Push toward correct tokens                   │
│                                                                  │
│  RL:     "Here's a response the model generated. Here's its score."   │
│          Loss = -log P(generated tokens) × advantage             │
│          Gradient = Push toward/away based on score              │
│                                                                  │
│  Key difference: RL doesn't know what's "correct" - it only     │
│  knows what scored well or poorly, and adjusts accordingly.     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

#### Code Comparison: SFT vs RL Training Step

Here's how the actual PyTorch code would look, to drive home the similarity:

```python
# ═══════════════════════════════════════════════════════════════════
# SFT TRAINING STEP
# ═══════════════════════════════════════════════════════════════════

def sft_training_step(model, prompt, ground_truth_response):
    """
    Standard supervised fine-tuning.
    We KNOW the correct response.
    """
    # Forward pass: get log probabilities for each token position
    logits = model(prompt + ground_truth_response)
    log_probs = F.log_softmax(logits, dim=-1)

    # For each position, get the log prob of the CORRECT next token
    # ground_truth_tokens[t] tells us what token SHOULD come next
    token_log_probs = log_probs[range(T), ground_truth_tokens]

    # Loss: negative mean log probability (cross-entropy)
    # We want to MAXIMIZE log prob, so MINIMIZE negative log prob
    loss = -token_log_probs.mean()

    # All tokens weighted equally (implicitly weight = 1)
    return loss


# ═══════════════════════════════════════════════════════════════════
# RL (REINFORCE) TRAINING STEP
# ═══════════════════════════════════════════════════════════════════

def rl_training_step(model, prompt, reward_model, baseline):
    """
    Reinforcement learning with policy gradient.
    We DON'T know the correct response - we sample and score.
    """
    # Step 1: SAMPLE a response from the model (not given to us!)
    sampled_response, sampled_tokens = model.generate(prompt)

    # Step 2: Get log probabilities for the tokens we SAMPLED
    logits = model(prompt + sampled_response)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[range(T), sampled_tokens]  # Our samples, not ground truth!

    # Step 3: Score the response with reward model
    reward = reward_model(prompt, sampled_response)

    # Step 4: Compute advantage (how much better than average?)
    advantage = reward - baseline  # e.g., baseline = running average of rewards

    # Step 5: RL Loss = -(sum of log probs) × advantage
    # If advantage > 0: we want to INCREASE these log probs → loss is negative → gradient descent increases them
    # If advantage < 0: we want to DECREASE these log probs → loss is positive → gradient descent decreases them
    loss = -(token_log_probs.sum()) * advantage

    # Key difference: tokens are weighted by advantage, not all equal!
    return loss


# ═══════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE COMPARISON
# ═══════════════════════════════════════════════════════════════════

"""
                        SFT                         RL
                        ───                         ──
Tokens:          ground_truth_tokens          sampled_tokens
                 (given to us)                (model generates)

Log probs of:    correct tokens              sampled tokens

Weighting:       1.0 (all equal)             advantage (varies)

Loss:            -mean(log_probs)            -sum(log_probs) × advantage

Gradient effect: Always push toward          Push toward if advantage > 0
                 correct tokens              Push away if advantage < 0
"""
```

**The punchline**: RL training is almost identical to SFT training! The only differences are:

1. We sample tokens instead of using ground truth
2. We weight the loss by the advantage instead of treating all tokens equally

---

### 1.4 Value Functions vs Reward Models

This is a crucial distinction that often causes confusion:

#### Reward Model $r_\phi(x, y)$

- **Purpose**: Provides the "ground truth" reward signal for training
- **Training**: Supervised learning on human preference data (Bradley-Terry)
- **Input**: Complete prompt-response pair
- **Output**: Scalar reward score
- **Frozen during RL**: Yes (typically)
- **Analogy**: The "judge" that scores responses

#### Value Function $V^\pi(s)$

- **Purpose**: Predicts _expected_ future reward to reduce variance in policy gradients
- **Training**: TD learning or Monte Carlo estimation during RL
- **Input**: Current state (partial generation)
- **Output**: Expected cumulative reward from this state
- **Updated during RL**: Yes, continuously
- **Analogy**: The "critic" that estimates how good the current situation is

#### Q-Function $Q^\pi(s, a)$

- **Purpose**: Predicts expected return after taking specific action
- **Relationship**: $Q^\pi(s, a) = r(s, a) + \gamma \mathbb{E}_{s'}[V^\pi(s')]$
- **In LLM context**: Estimates expected reward after generating a specific token

**Key Insight**: The reward model gives actual rewards; value functions _predict_ those rewards to make training more stable.

```
┌─────────────────────────────────────────────────────────────────┐
│                     How They Work Together                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Prompt x ──► Policy π_θ ──► Response y ──► Reward Model ──► r │
│                    │                              │              │
│                    ▼                              │              │
│            Value Function                         │              │
│            V(s) predicts                          │              │
│            expected r                             │              │
│                    │                              │              │
│                    ▼                              │              │
│            Advantage A = r - V(s)  ◄──────────────┘              │
│                    │                                             │
│                    ▼                                             │
│            Policy Gradient Update                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Token-Level vs Trajectory-Level Actions

**Common question:** _Is the action the entire rollout or is it each token? Do we need to compute Q and V at each token?_

**Answer:** It depends on the algorithm, but here is the typical setup:

#### Trajectory-Level (Most Common in LLM RLHF)

In methods like GRPO, RLOO, and standard PPO for LLMs:

1. **Action Definition**: The entire response $y = (y_1, \ldots, y_T)$ is treated as one action
2. **Reward**: Given only at the end of the sequence
3. **Advantage**: Computed per-trajectory, not per-token
4. **Rollout Completion**: Yes, we let the model finish generating before computing advantages

```python
# Pseudocode for trajectory-level approach
for prompt in batch:
    response = generate_complete_response(policy, prompt)  # Full rollout
    reward = reward_model(prompt, response)  # Single scalar
    advantage = compute_advantage(reward)  # Per-trajectory
```

#### Token-Level (More Sophisticated)

Recent work explores token-level RL:

1. **Action Definition**: Each token $y_t$ is a separate action
2. **Reward**: Can be dense (reward at each token) or sparse (only at end, propagated back)
3. **Q and V**: Computed at each token position
4. **Advantage**: $A_t = Q(s_t, a_t) - V(s_t)$ at each timestep

**Token-Level Challenges**:

- Requires value function that can estimate reward from partial sequences
- Credit assignment: Which tokens contributed to final reward?
- Computational overhead: Need forward passes at each position

**Practical Hybrid Approach** (GAE in PPO):

Even with trajectory-level rewards, we can compute token-level advantages using GAE:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$

Where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

This propagates the final reward backward through the value function to give credit to each token.

---

## 2. Reward Model Training

### 2.1 Bradley-Terry Model

The **Bradley-Terry (BT) model** is the standard approach for converting pairwise human preferences into a reward model.

#### Setup

Given a prompt $x$ and two responses $y_w$ (preferred/winner) and $y_l$ (rejected/loser), we want to learn a reward function $r_\phi(x, y)$.

#### Bradley-Terry Preference Model

The probability that $y_w$ is preferred over $y_l$ is modeled as:

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l)) = \frac{\exp(r_\phi(x, y_w))}{\exp(r_\phi(x, y_w)) + \exp(r_\phi(x, y_l))}$$

Where $\sigma$ is the sigmoid function.

**Intuition**: The reward difference determines preference probability through a logistic model.

#### Training Objective

We minimize the negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))\right]$$

This is essentially **binary classification**: given a pair, predict which is preferred.

#### Implementation Notes

```python
def reward_model_loss(reward_model, x, y_w, y_l):
    r_w = reward_model(x, y_w)  # Scalar reward for preferred
    r_l = reward_model(x, y_l)  # Scalar reward for rejected
    loss = -torch.log(torch.sigmoid(r_w - r_l))
    return loss.mean()
```

### 2.2 Practical Considerations

**Architecture**: Typically a pretrained LLM with a scalar head on top of the final hidden state.

**Order Consistency**: The reward model only needs to preserve correct _rankings_, not absolute values. Any monotonic transformation of a correct reward model is still correct.

**Limitations**:

- $O(n^2)$ complexity for pairwise comparisons
- Distribution shift: May not generalize to outputs far from training distribution
- Reward hacking: Policy may find adversarial outputs that score high but are low quality

**Alternatives**:

- **Pointwise Reward Models**: Predict scalar scores directly (simpler but less accurate)
- **Process Reward Models (PRMs)**: Give rewards at intermediate steps, not just the end

---

## 3. REINFORCE Algorithm

### 3.1 Core Algorithm

REINFORCE is the simplest policy gradient algorithm, directly applying the policy gradient theorem.

#### Algorithm

1. Sample trajectory $\tau = (s_0, a_0, r_0, \ldots, s_T, a_T, r_T) \sim \pi_\theta$
2. Compute return $R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ for each timestep
3. Update policy:

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t$$

#### For LLM Training (Trajectory-Level)

In the LLM context with trajectory-level rewards:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(y | x) \cdot r(x, y)\right]$$

Where $\log \pi_\theta(y | x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})$ (sum of token log-probs).

**No "old" policy in REINFORCE.** REINFORCE is **strictly on-policy**: you sample trajectories from the current policy $\pi_\theta$, and the gradient uses the same policy's log probs $\log \pi_\theta(a_t|s_t)$. There is no stored "old" policy and no importance-sampling ratio. Each batch is generated by the policy you're updating, and you typically use that batch for exactly one gradient step (then throw it away and sample again). So there are no "off-policy" generations in the usual REINFORCE formulation—everything is generated by the current $\pi_\theta$. Contrast this with PPO below, which keeps $\pi_{\text{old}}$ and reuses the same rollouts for multiple updates via the clipped ratio.

### 3.2 Variance Reduction Techniques

REINFORCE suffers from high variance. Key techniques to address this:

#### Baseline Subtraction

Replace $R_t$ with advantage $A_t = R_t - b(s_t)$:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot (R_t - b(s_t))\right]$$

This is **unbiased** as long as $b$ doesn't depend on the action. Common choices:

- **Constant baseline**: Running average of rewards
- **Learned baseline**: Value function $V(s)$

#### Reward Normalization

Normalize rewards across the batch:

$$\hat{R} = \frac{R - \mu_R}{\sigma_R}$$

This stabilizes training but introduces some bias.

### 3.3 REINFORCE++ and RLOO

#### REINFORCE++

A recent improvement using **global advantage normalization**:

$$\hat{A}_i = \frac{R_i - \mu_{\text{batch}}}{\sigma_{\text{batch}}}$$

Normalizes across the entire batch (not just per-prompt groups), providing more stable, asymptotically unbiased estimates.

**Key insight**: Bias vanishes as batch size increases, unlike per-prompt normalization.

#### RLOO (REINFORCE Leave-One-Out)

Uses a **leave-one-out baseline** for unbiased advantage estimation:

For $K$ samples per prompt, the advantage for sample $i$:

$$\hat{A}_i = R_i - \frac{1}{K-1}\sum_{j \neq i} R_j$$

Equivalently:
$$\hat{A}_i = \frac{K}{K-1}(R_i - \bar{R})$$

**Why unbiased?** The baseline excludes the current sample, so reward $R_i$ is independent of its baseline.

**KL Regularization in RLOO**:

$$R'_i = R_i - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

---

## 4. Proximal Policy Optimization (PPO)

PPO is the workhorse algorithm for RLHF. It builds on REINFORCE with several improvements.

**Old vs new policy (and why PPO can reuse data).** Unlike REINFORCE, PPO explicitly keeps an **old** policy $\pi_{\theta_{\text{old}}}$ (or $\pi_{\text{old}}$): rollouts are collected with that policy, and we store the **old log probs** for each action. When we update $\theta$, we get a **new** policy $\pi_\theta$. The objective uses the **ratio** $r_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$. Because the ratio corrects for the change in policy, the same batch of rollouts can be used for **multiple gradient steps** (e.g. 4–8 epochs over the same data) without the gradient becoming invalid—as long as we clip the ratio so $\pi_\theta$ doesn't drift too far from $\pi_{\text{old}}$. So in PPO there is a clear notion of "data generated under the old policy" and "updates applied to the new policy," with off-policy-style correction via the ratio and clipping. In REINFORCE there is no such split: data and update are both for the same policy.

### 4.1 Trust Region Methods

**Problem**: Large policy updates can be catastrophic—the policy might move to a bad region and never recover.

**Solution**: Constrain how much the policy can change per update.

#### TRPO (Trust Region Policy Optimization)

Solves a constrained optimization:

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{\text{old}}}(a|s)} A^{\pi_{\theta_{\text{old}}}}(s, a)\right]$$

Subject to: $D_{\text{KL}}(\pi_{\theta_{\text{old}}} \| \pi_\theta) \leq \delta$

This is computationally expensive (requires conjugate gradient, Fisher information matrix).

### 4.2 PPO Clipped Objective

PPO simplifies trust regions using a **clipped surrogate objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

**Where do the log probs go?** The ratio $r_t$ is built from log probabilities. For a single token, $r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} = \exp\bigl(\log \pi_\theta(a_t|s_t) - \log \pi_{\text{old}}(a_t|s_t)\bigr)$. For an entire response we need the **trajectory** probability ratio. Under the usual autoregressive factorization, $\pi(y|x) = \prod_{t=1}^T \pi(y_t | x, y_{<t})$, so in log space we get a **sum over tokens**:

$$\log \pi_\theta(y|x) = \sum_{t=1}^{T} \log \pi_\theta(y_t | x, y_{<t})$$

The **trajectory ratio** used in practice is then $\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} = \exp\left(\sum_{t=1}^T \bigl[\log \pi_\theta(y_t|s_t) - \log \pi_{\text{old}}(y_t|s_t)\bigr]\right)$. So we do sum log probs over all tokens; that sum appears inside the exponent that gives the ratio. When using **per-token** advantages (e.g. GAE), the objective is a sum over $t$ of $\min(r_t A_t, \text{clip}(r_t) A_t)$ where each $r_t$ is the per-token ratio; when using a **single advantage per trajectory** (e.g. one reward at the end), the ratio is the trajectory ratio above (one scalar per response), and we still compute it from the sum of log-prob differences over tokens.

---

#### What is "clip"? (Plain English Explanation)

The `clip` function is just a way to bound a number within a range:

```
clip(x, lower, upper) =
    - If x < lower: return lower
    - If x > upper: return upper
    - Otherwise: return x

Examples:
    clip(0.5, 0.8, 1.2) = 0.8   (0.5 is below 0.8, so return 0.8)
    clip(1.0, 0.8, 1.2) = 1.0   (1.0 is in range, so return 1.0)
    clip(1.5, 0.8, 1.2) = 1.2   (1.5 is above 1.2, so return 1.2)
```

**In PPO**: $\text{clip}(r_t, 1-\epsilon, 1+\epsilon)$ with $\epsilon = 0.2$ means:

- Bound $r_t$ to be between 0.8 and 1.2
- This limits how much the policy can change in one update

---

#### What is $r_t$? (The Probability Ratio)

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$$

**In plain English**: "How much more (or less) likely is this token under the new policy compared to the old policy?"

| Value of $r_t$ | Meaning                                                       |
| -------------- | ------------------------------------------------------------- |
| $r_t = 1.0$    | New policy assigns same probability as old policy (no change) |
| $r_t = 1.5$    | New policy is 50% more likely to produce this token           |
| $r_t = 0.5$    | New policy is 50% less likely to produce this token           |
| $r_t = 2.0$    | New policy is 2x more likely to produce this token            |

---

#### The Full PPO Objective in Plain English

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 0.8, 1.2) \cdot A_t\right)\right]$$

**What this says in English**:

1. Compute $r_t \cdot A_t$: "How much do we want to change, weighted by advantage"
2. Compute $\text{clip}(r_t, 0.8, 1.2) \cdot A_t$: "Same thing, but cap the ratio at 0.8-1.2"
3. Take the minimum of these two
4. This is our loss (we maximize it, so good actions get reinforced)

**Why the minimum?** It's a pessimistic bound that prevents too-large updates:

---

#### How Clipping Works: Detailed Walkthrough

**Case 1: Good action (A > 0) and we want to increase its probability**

```
Situation: Token "4" was good (advantage = +0.5)
           Old policy: P("4") = 0.3
           New policy: P("4") = 0.45
           Ratio: r = 0.45/0.3 = 1.5 (50% increase)

Without clipping:
    Loss = r × A = 1.5 × 0.5 = 0.75
    Gradient keeps pushing up P("4")

With clipping (ε = 0.2):
    clipped_r = clip(1.5, 0.8, 1.2) = 1.2  (capped!)
    Loss = min(1.5 × 0.5, 1.2 × 0.5) = min(0.75, 0.6) = 0.6

    The gradient is now ZERO for further increases!
    We've already increased P("4") by 20%, that's enough for one update.
```

**Case 2: Bad action (A < 0) and we want to decrease its probability**

```
Situation: Token "5" was bad (advantage = -0.8)
           Old policy: P("5") = 0.2
           New policy: P("5") = 0.08
           Ratio: r = 0.08/0.2 = 0.4 (60% decrease)

Without clipping:
    Loss = r × A = 0.4 × (-0.8) = -0.32
    Gradient keeps pushing down P("5")

With clipping (ε = 0.2):
    clipped_r = clip(0.4, 0.8, 1.2) = 0.8  (raised to minimum!)
    Loss = min(0.4 × (-0.8), 0.8 × (-0.8)) = min(-0.32, -0.64) = -0.64

    Wait, min gives -0.64 which is the clipped version.
    But since A < 0, min actually picks the LESS negative value...

    Actually, for A < 0:
    min(r × A, clip(r) × A) with r < 1 gives the clipped version
    This stops us from decreasing probability too much.
```

**The visual intuition**:

```
                    Probability Ratio (r)
                           │
         Too much          │          Too much
         decrease          │          increase
           ◄───────────────┼───────────────►
                           │
    ┌──────────────────────┼──────────────────────┐
    │      CLIP ZONE       │       CLIP ZONE      │
    │    (no gradient)     │     (no gradient)    │
    │◄─────────────────────┼─────────────────────►│
    0.8                   1.0                    1.2
    │                      │                      │
    │     ACTIVE ZONE      │                      │
    │   (gradient flows)   │                      │
    └──────────────────────┴──────────────────────┘

    When r is in [0.8, 1.2]: Normal gradient, policy updates
    When r goes outside: Gradient becomes zero, policy stops changing
```

---

**Summary in one sentence**: Clipping limits how much any token's probability can increase or decrease to at most 20% per update step, no matter how good or bad the advantage is.

### 4.3 Generalized Advantage Estimation (GAE)

GAE provides a smooth tradeoff between bias and variance in advantage estimation.

#### TD Error

$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This is a biased but low-variance estimate of advantage.

#### GAE Formula

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V$$

In practice (finite horizon):

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \ldots + (\gamma\lambda)^{T-t}\delta_T$$

#### Lambda Parameter

| $\lambda$         | Behavior                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------ |
| $\lambda = 0$     | One-step TD: $\hat{A}_t = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (low variance, high bias) |
| $\lambda = 1$     | Monte Carlo: $\hat{A}_t = \sum_{l=0}^{T-t} \gamma^l r_{t+l} - V(s_t)$ (high variance, low bias)  |
| $0 < \lambda < 1$ | Interpolation (typically $\lambda = 0.95$)                                                       |

### 4.4 Actor-Critic Architecture

PPO uses two networks (often sharing a backbone):

#### Actor (Policy Network)

- Input: State (prompt + generated tokens)
- Output: Probability distribution over next token
- Training: Policy gradient with clipped objective

#### Critic (Value Network)

- Input: State
- Output: Scalar value estimate $V(s)$
- Training: Minimize MSE between predicted and actual returns

$$L^V(\phi) = \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right]$$

Where $R_t$ is the empirical return (Monte Carlo) or bootstrapped estimate.

#### Combined PPO Objective

$$L^{\text{PPO}}(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 L^V(\phi) + c_2 S[\pi_\theta]$$

Where:

- $c_1$: Value loss coefficient
- $c_2$: Entropy bonus coefficient (encourages exploration)
- $S[\pi_\theta]$: Entropy of the policy

---

## 5. A2C: Advantage Actor-Critic

A2C is the synchronous version of A3C (Asynchronous Advantage Actor-Critic).

### Core Idea

Combine policy gradient (actor) with value function baseline (critic) to reduce variance:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t | s_t) \cdot A(s_t, a_t)\right]$$

Where $A(s_t, a_t) = Q(s_t, a_t) - V(s_t)$

### Simplified Advantage

Using Bellman equation:
$$A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

This only requires the value function $V$, not $Q$.

### A2C vs PPO

| Aspect            | A2C                  | PPO             |
| ----------------- | -------------------- | --------------- |
| Trust region      | None                 | Clipped ratio   |
| Stability         | Lower                | Higher          |
| Sample efficiency | Lower                | Higher          |
| Implementation    | Simpler              | More complex    |
| Updates           | Single gradient step | Multiple epochs |

---

## 6. Group Relative Policy Optimization (GRPO)

GRPO is the algorithm behind DeepSeek-R1 and is increasingly popular for LLM training.

### 6.1 Core Innovation

**Key Insight**: Replace the expensive critic network with group-based normalization.

Instead of learning $V(s)$ to estimate expected reward, GRPO:

1. Samples multiple responses per prompt
2. Uses the group's average reward as the baseline

This eliminates the need for a separate value network, saving memory and compute.

### 6.2 Mathematical Formulation

#### Sampling

For each prompt $x$, sample $G$ responses: $\{y_1, y_2, \ldots, y_G\} \sim \pi_\theta(\cdot | x)$

#### Reward Computation

Score each response: $r_i = r(x, y_i)$ for $i = 1, \ldots, G$

#### Group-Relative Advantage

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}$$

Where:

- $\mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j$ (group mean)
- $\sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}$ (group std)

---

**Plain English: What is this computing?**

This is just a **z-score** (standardization) of the rewards:

```
advantage = (reward - average_reward) / standard_deviation
```

**Concrete example**:

```
Prompt: "Write a function to add two numbers"

Generated 8 responses with rewards:
   Response 1: reward = 0.9  (correct, well-formatted)
   Response 2: reward = 0.8  (correct, okay format)
   Response 3: reward = 0.7  (correct, verbose)
   Response 4: reward = 0.6  (correct, minor issues)
   Response 5: reward = 0.3  (has a bug)
   Response 6: reward = 0.2  (wrong approach)
   Response 7: reward = 0.5  (partially correct)
   Response 8: reward = 0.4  (has syntax error)

Step 1: Compute mean
   μ = (0.9+0.8+0.7+0.6+0.3+0.2+0.5+0.4) / 8 = 0.55

Step 2: Compute standard deviation
   σ ≈ 0.23

Step 3: Compute advantages (z-scores)
   Response 1: (0.9 - 0.55) / 0.23 = +1.52  ← "Much better than average"
   Response 2: (0.8 - 0.55) / 0.23 = +1.09  ← "Better than average"
   Response 3: (0.7 - 0.55) / 0.23 = +0.65  ← "Somewhat better"
   Response 4: (0.6 - 0.55) / 0.23 = +0.22  ← "Slightly better"
   Response 5: (0.3 - 0.55) / 0.23 = -1.09  ← "Worse than average"
   Response 6: (0.2 - 0.55) / 0.23 = -1.52  ← "Much worse than average"
   Response 7: (0.5 - 0.55) / 0.23 = -0.22  ← "Slightly worse"
   Response 8: (0.4 - 0.55) / 0.23 = -0.65  ← "Somewhat worse"
```

**What happens next**:

- Responses with positive advantage → tokens get reinforced (more likely)
- Responses with negative advantage → tokens get suppressed (less likely)

**Why divide by standard deviation?**

If all responses in a group have similar rewards (σ is small), the differences get amplified. If rewards vary a lot (σ is large), the differences get dampened. This keeps the gradient magnitude consistent across different prompts.

#### GRPO Objective

$$L^{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{G} \min\left(r_t^{(i)}(\theta) \hat{A}_i, \text{clip}(r_t^{(i)}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Where $r_t^{(i)}(\theta) = \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{\text{old}}}(y_i | x)}$

---

#### Plain English: What Does the GRPO Objective Actually Mean?

Let's break this scary formula into pieces:

**The full objective has two parts**:

$$\underbrace{\mathbb{E}\left[\sum_{i=1}^{G} \min\left(r_t^{(i)} \hat{A}_i, \text{clip}(r_t^{(i)}, 0.8, 1.2) \hat{A}_i\right)\right]}_{\text{Part 1: PPO-style clipped policy update}} - \underbrace{\beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})}_{\text{Part 2: KL penalty}}$$

---

**Part 1: The clipped policy update (exactly like PPO)**

This is identical to PPO's clipped objective. In English:

```
For each response in the group:
    1. Compute advantage: How much better/worse than the group average?
    2. Compute ratio: How much more/less likely under new policy vs old?
    3. Apply clipping: Don't let the ratio go outside [0.8, 1.2]
    4. Multiply advantage × ratio (clipped if needed)
    5. Sum across all responses in the group
```

---

**Part 2: The KL penalty**

```
Compute: How different is our policy from the original model?
Multiply by β (a small number like 0.01 or 0.1)
Subtract from the objective (penalize divergence)
```

---

**The complete GRPO training step in plain English**:

```
GRPO Training Step:
───────────────────

1. SAMPLE: Generate G=16 responses for the same prompt
   "What is 2+2?" → ["4", "The answer is 4", "2+2=4", "It's 4", ...]

2. SCORE: Get reward for each response
   Rewards: [1.0, 0.9, 0.95, 0.85, 0.2, 0.8, ...]

3. COMPUTE ADVANTAGE: Normalize within the group
   Mean reward: 0.75
   Std reward: 0.25
   Advantages: [(1.0-0.75)/0.25, (0.9-0.75)/0.25, ...] = [1.0, 0.6, 0.8, 0.4, -2.2, 0.2, ...]

4. FOR EACH RESPONSE:
   If advantage > 0 (better than average):
       → Increase probability of all tokens in this response
       → But clip so we don't increase by more than 20%

   If advantage < 0 (worse than average):
       → Decrease probability of all tokens in this response
       → But clip so we don't decrease by more than 20%

5. ADD KL PENALTY:
   → Also penalize if policy drifts too far from reference

6. UPDATE WEIGHTS:
   → Take gradient step on this combined objective
```

---

**Why GRPO works without a value function**:

The key trick is step 3: instead of learning a value function $V(s)$ to estimate "how good is this state?", GRPO just:

1. Generates multiple responses
2. Uses the **average reward of the group** as the baseline
3. Responses above average → reinforce them
4. Responses below average → suppress them

This is simpler and cheaper than maintaining a separate value network!

### 6.3 KL Divergence Penalty

**Common question:** _Is the reference always the base model or does it update? How is KL computed?_

#### Reference Policy

The reference policy $\pi_{\text{ref}}$ is typically:

- **Fixed**: The initial SFT model, frozen throughout training
- **Not updated**: It serves as an anchor to prevent the policy from drifting too far

Some variants use **iterative reference updates** where $\pi_{\text{ref}}$ is periodically updated to the current policy, but this is less common.

#### KL Computation

For a response $y = (y_1, \ldots, y_T)$:

$$D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \sum_{t=1}^{T} \sum_{v \in \mathcal{V}} \pi_\theta(v | x, y_{<t}) \log \frac{\pi_\theta(v | x, y_{<t})}{\pi_{\text{ref}}(v | x, y_{<t})}$$

**In practice** (Monte Carlo estimate using generated tokens):

$$\hat{D}_{\text{KL}} = \sum_{t=1}^{T} \left[\log \pi_\theta(y_t | x, y_{<t}) - \log \pi_{\text{ref}}(y_t | x, y_{<t})\right]$$

---

#### Plain English: What Are These Formulas Actually Computing?

**The scary first formula** (theoretical KL):

```
For each token position in the response:
    For EVERY word in the vocabulary (50,000+ words):
        1. Get probability from current policy: P_policy("word")
        2. Get probability from reference model: P_ref("word")
        3. Compute: P_policy × log(P_policy / P_ref)
        4. Add to running sum
```

This asks: "Across the entire vocabulary, how different are the two probability distributions?"

**Why we don't compute this**: Summing over 50,000 vocabulary items at every token position is expensive.

---

**The practical second formula** (what we actually compute):

```
For each token that was actually generated:
    1. Get log-prob from current policy: log P_policy(token)
    2. Get log-prob from reference: log P_ref(token)
    3. Compute difference: log P_policy(token) - log P_ref(token)
    4. Add to running sum
```

**In even simpler terms**:

| What we compute              | Plain English                                                             |
| ---------------------------- | ------------------------------------------------------------------------- |
| $\log \pi_\theta(y_t)$       | "How likely does our current model think this token is?"                  |
| $\log \pi_{\text{ref}}(y_t)$ | "How likely did the original model think this token is?"                  |
| Difference                   | "Is our model MORE confident (+) or LESS confident (-) about this token?" |
| Sum over all tokens          | "Overall, how much has our model's confidence changed from the original?" |

---

**Concrete Example**:

```
Response: "The answer is 4"

Token 1: "The"
    Policy log-prob: -1.2      (P ≈ 30%)
    Reference log-prob: -1.3   (P ≈ 27%)
    Difference: -1.2 - (-1.3) = +0.1
    → Policy is slightly MORE confident about "The"

Token 2: "answer"
    Policy log-prob: -2.5      (P ≈ 8%)
    Reference log-prob: -2.0   (P ≈ 14%)
    Difference: -2.5 - (-2.0) = -0.5
    → Policy is LESS confident about "answer"

Token 3: "is"
    Policy log-prob: -0.8      (P ≈ 45%)
    Reference log-prob: -0.9   (P ≈ 41%)
    Difference: +0.1

Token 4: "4"
    Policy log-prob: -0.3      (P ≈ 74%)
    Reference log-prob: -1.5   (P ≈ 22%)
    Difference: -0.3 - (-1.5) = +1.2
    → Policy is MUCH more confident about "4" (this is what RL trained!)

Total KL = 0.1 + (-0.5) + 0.1 + 1.2 = 0.9

This KL penalty of 0.9 says: "The policy has diverged from the reference,
mainly because it's become very confident about outputting '4'"
```

---

**Why do we penalize this?**

If KL gets too high, the policy has drifted far from the original model:

- It might have "forgotten" general language capabilities
- It might be overfitting to the reward model
- It might have found a weird exploit (reward hacking)

The KL penalty says: "Yes, optimize for reward, but don't become too different from the original model."

---

#### Why No Extra Scaling Factor?

A common question is: "Shouldn't the log-prob difference be multiplied by the probability $\pi_\theta(y_t)$?" The answer is **no**, because the $\pi_\theta$ weighting is already accounted for by the act of sampling. The true KL is $\mathbb{E}_{y \sim \pi_\theta}[\log \frac{\pi_\theta}{\pi_{\text{ref}}}]$. When we sample $y_t$ from $\pi_\theta$, each sample already "carries" the $\pi_\theta(y_t)$ weight implicitly—that is how Monte Carlo estimation works. We just compute $\log \pi_\theta(y_t) - \log \pi_{\text{ref}}(y_t)$ per sampled token, sum over tokens, and that is an unbiased estimate of the KL.

#### KL Estimator Variants (k1, k2, k3)

The formula above ($\log \pi_\theta - \log \pi_{\text{ref}}$) is known as the **k1 estimator** from Schulman's "Approximating KL Divergence" blog post. It is unbiased but has high variance—individual per-token values can be **negative** even though the true KL is always $\geq 0$. In practice, three estimators are commonly discussed:

| Estimator | Formula (where $r = \pi_\theta / \pi_{\text{ref}}$) | Bias? | Variance | Always $\geq 0$? |
|-----------|------------------------------------------------------|-------|----------|-------------------|
| **k1**    | $\log r$                                             | No    | High     | No                |
| **k2**    | $\frac{1}{2}(\log r)^2$                              | Yes   | Low      | Yes               |
| **k3**    | $(r - 1) - \log r$                                   | No    | Low      | Yes               |

**k3** is particularly attractive: it is unbiased (via a control-variate argument) and always non-negative. It is used in many GRPO implementations, while PPO implementations commonly use **k1**.

#### Implementation

```python
def compute_kl_penalty(policy_logprobs, ref_logprobs, estimator="k1"):
    """
    policy_logprobs: log π_θ(y_t | context) for each token
    ref_logprobs: log π_ref(y_t | context) for each token

    Example:
        policy_logprobs = [-1.2, -2.5, -0.8, -0.3]  # 4 tokens
        ref_logprobs    = [-1.3, -2.0, -0.9, -1.5]  # 4 tokens
    """
    log_ratio = policy_logprobs - ref_logprobs  # Per-token log(π_θ/π_ref)

    if estimator == "k1":
        # Unbiased, high variance, can be negative per-token
        kl = log_ratio
    elif estimator == "k3":
        # Unbiased, lower variance, always >= 0
        ratio = torch.exp(log_ratio)
        kl = (ratio - 1) - log_ratio
    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # kl (k1) = [0.1, -0.5, 0.1, 1.2]  → sum = 0.9
    # kl (k3) = [0.005, 0.132, 0.005, 0.515] → sum = 0.657 (always non-negative per token)
    return kl.sum()
```

#### Why KL Penalty?

1. **Prevents reward hacking**: Without KL, policy might find adversarial outputs
2. **Preserves capabilities**: Keeps model close to pretrained distribution
3. **Regularization**: Acts like Bayesian prior, preventing overfitting to reward model

### 6.4 DeepSeek-R1 Implementation Details

#### Training Pipeline

1. **Sampling**: Generate $G = 16$ responses per prompt at temperature 0.7-1.0
2. **Reward Scoring**: Use rule-based rewards (accuracy + format)
3. **Advantage**: Normalize within group
4. **Update**: Clipped policy gradient with KL penalty

#### Reward Function (R1-Zero)

For reasoning tasks:

- **Accuracy reward**: Binary (correct/incorrect)
- **Format reward**: Encourages using `<think>...</think>` tags

This simple rule-based reward achieved 71% on AIME 2024 (from 15.6% baseline).

---

## 7. Direct Preference Optimization (DPO)

DPO is an "RL-free" method that bypasses explicit reward modeling.

### 7.1 Theoretical Foundation

**Key Insight**: The optimal policy for KL-regularized reward maximization has a closed form:

$$\pi^*(y | x) = \frac{\pi_{\text{ref}}(y | x) \exp(r(x, y) / \beta)}{Z(x)}$$

Where $Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp(r(x, y) / \beta)$ is the partition function.

**Rearranging**:

$$r(x, y) = \beta \log \frac{\pi^*(y | x)}{\pi_{\text{ref}}(y | x)} + \beta \log Z(x)$$

This means: **if we can learn the optimal policy, we implicitly have a reward model**.

### 7.2 DPO Loss Derivation

Starting from Bradley-Terry:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Substituting the implicit reward:

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi^*(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)$$

Note: $\log Z(x)$ cancels out!

#### DPO Loss

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

#### Simplified Form

Define: $\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}$

Then: $\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l))\right]$

### 7.3 Implicit Reward Model

The trained policy implicitly defines a reward:

$$r(x, y) = \beta \log \frac{\pi_\theta(y | x)}{\pi_{\text{ref}}(y | x)}$$

This can be extracted and used for scoring, showing that **the language model is secretly a reward model**.

### Advantages of DPO

1. **No sampling during training**: Purely supervised on preference pairs
2. **No reward model**: Eliminates separate RM training
3. **No value function**: No critic needed
4. **Simpler implementation**: Standard classification loss

### Limitations

1. **Offline only**: Can't do online/iterative improvement
2. **Data efficiency**: May need more preference data
3. **Over-optimization**: Can overfit to preference data

---

## 8. Identity Preference Optimization (IPO)

### 8.1 DeepMind's Approach

IPO (from "A General Theoretical Paradigm to Understand Learning from Human Feedback") addresses theoretical issues with DPO.

#### The ΨPO Framework

IPO is a special case of ΨPO, which generalizes preference optimization:

$$\mathcal{L}_{\Psi\text{PO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)}\left[\Psi\left(\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right)\right]$$

Different choices of $\Psi$ give different algorithms:

- **DPO**: $\Psi(u) = -\log\sigma(\beta u)$
- **IPO**: $\Psi(u) = (u - 1/2\tau)^2$ (identity-based regularization)

### 8.2 Comparison with DPO

| Aspect            | DPO                               | IPO                            |
| ----------------- | --------------------------------- | ------------------------------ |
| Preference Model  | Bradley-Terry (pointwise rewards) | Direct pairwise preferences    |
| Over-training     | Prone to over-optimization        | Robust to over-training        |
| Theoretical Basis | Reward model → RL reduction       | Direct preference optimization |
| Regularization    | Through implicit reward           | Built into objective           |

#### Key Theoretical Difference

- **DPO** assumes preferences can be reduced to pointwise rewards (Bradley-Terry)
- **IPO** works directly with pairwise preferences, avoiding this potentially lossy reduction

**When preferences are noisy or don't fit Bradley-Terry**, IPO may perform better.

---

## 9. Distributed Training in RL

### 9.1 Architecture Paradigms

#### Single-Controller Architecture

**Example**: LlamaRL

```
┌─────────────────────────────────────────────────────────┐
│                    Controller                           │
│   (Coordinates all workers, manages global state)       │
├─────────────────────────────────────────────────────────┤
│                         │                               │
│    ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐   │
│    │Worker 1│   │Worker 2│   │Worker 3│   │Worker N│   │
│    └────────┘   └────────┘   └────────┘   └────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Pros**: Simpler coordination, easier debugging
**Cons**: Controller can become bottleneck

#### Multi-Controller Architecture

**Example**: DistFlow

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 1 │◄─►│Worker 2 │◄─►│Worker 3 │◄─►│Worker N │
│+Control │   │+Control │   │+Control │   │+Control │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
     │             │             │             │
     └─────────────┴─────────────┴─────────────┘
                   (Peer-to-peer)
```

**Pros**: Near-linear scaling, no centralized bottleneck
**Cons**: More complex coordination

### 9.2 Key Components: Samplers, Trainers, Learners, Actors, Critics

#### Actors / Samplers

**Role**: Generate rollouts (trajectories) from the current policy

```python
class Actor:
    def __init__(self, policy):
        self.policy = policy

    def generate_trajectory(self, prompt):
        response = sample_from_policy(self.policy, prompt)
        return (prompt, response)
```

- Run inference on the policy network
- May be distributed across many GPUs for throughput
- Typically use the latest policy weights

#### Critics

**Role**: Evaluate states/actions to compute value estimates

```python
class Critic:
    def __init__(self, value_network):
        self.V = value_network

    def estimate_value(self, state):
        return self.V(state)
```

- Compute $V(s)$ or $Q(s, a)$ for advantage calculation
- In critic-free methods (GRPO, RLOO), this role is eliminated

#### Learners / Trainers

**Role**: Update model parameters based on collected experience

```python
class Learner:
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer

    def update(self, batch):
        loss = compute_ppo_loss(self.policy, batch)
        loss.backward()
        self.optimizer.step()
```

- Compute gradients and update weights
- Synchronize weights across distributed actors
- Handle gradient accumulation, mixed precision, etc.

#### A2C/A3C Context

- **A2C (Advantage Actor-Critic)**: Synchronous—all actors complete, then single update
- **A3C (Asynchronous Advantage Actor-Critic)**: Actors update asynchronously

Modern distributed RL often uses **asynchronous off-policy** training where:

1. Actors generate data continuously
2. Learners consume data from a replay buffer
3. Weight synchronization happens periodically

---

## 10. Advanced Topics

### 10.1 On-Policy vs Off-Policy Training

#### On-Policy

**Definition**: Learn from data generated by the _current_ policy

**Examples**: REINFORCE, A2C, standard PPO

**Characteristics**:

- Data becomes stale immediately after policy update
- Requires fresh samples for each update
- Lower sample efficiency but simpler

#### Off-Policy

**Definition**: Learn from data generated by _any_ policy (including old versions)

**Examples**: Q-learning, SAC, off-policy PPO variants

**Characteristics**:

- Can reuse old data (replay buffer)
- Higher sample efficiency
- Requires importance sampling correction

### 10.2 Importance Sampling

When learning off-policy, we correct for the distribution mismatch:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_{\text{old}}}\left[\frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)} \nabla_\theta \log \pi_\theta(\tau) R(\tau)\right]$$

The ratio $\frac{\pi_\theta(\tau)}{\pi_{\text{old}}(\tau)}$ can have **high variance** over long trajectories.

**Solutions**:

- **Clipping** (PPO): Bound the ratio
- **Truncation**: Cap the ratio at some maximum
- **Per-step IS**: Apply ratio at each step, not entire trajectory

### 10.3 Process Reward Models vs Outcome Reward Models

#### Outcome Reward Models (ORMs)

- Reward given only at the end of the response
- Simpler to collect data for
- Harder for credit assignment

#### Process Reward Models (PRMs)

- Reward given at intermediate steps
- Better credit assignment
- Harder to collect labels (need step-by-step annotations)

**For reasoning/code tasks**, PRMs can significantly improve performance by rewarding good intermediate steps, not just correct final answers.

---

## 11. Algorithm Comparison Summary

| Algorithm       | Critic? | Reward Model | On/Off Policy | KL Regularization | Key Feature                  |
| --------------- | ------- | ------------ | ------------- | ----------------- | ---------------------------- |
| **REINFORCE**   | No      | Explicit     | On            | Optional          | Simplest policy gradient     |
| **RLOO**        | No      | Explicit     | On            | In reward         | Leave-one-out baseline       |
| **REINFORCE++** | No      | Explicit     | On            | In reward         | Global normalization         |
| **A2C**         | Yes     | Explicit     | On            | No                | Actor-critic baseline        |
| **PPO**         | Yes     | Explicit     | On            | Optional          | Clipped objective            |
| **GRPO**        | No      | Explicit     | On            | In loss           | Group-relative advantage     |
| **DPO**         | No      | Implicit     | N/A (offline) | Implicit          | Direct preference learning   |
| **IPO**         | No      | Implicit     | N/A (offline) | In loss           | Robust pairwise optimization |

### When to Use What

| Scenario                                 | Recommended Algorithm    |
| ---------------------------------------- | ------------------------ |
| Abundant compute, need maximum control   | PPO with critic          |
| Large-scale training, memory constrained | GRPO or RLOO             |
| Offline preference data only             | DPO or IPO               |
| Noisy preferences                        | IPO                      |
| Simple setup, quick iteration            | REINFORCE++              |
| Reasoning with verifiable rewards        | GRPO (as in DeepSeek-R1) |

---

## 12. Further Reading and Resources

### Papers

#### Core Algorithms

1. **REINFORCE**: Williams, R.J. "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning" (1992)

   - [Paper](https://link.springer.com/article/10.1007/BF00992696)

2. **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms" (2017)

   - [Paper](https://arxiv.org/abs/1707.06347)

3. **GAE**: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (2015)

   - [Paper](https://arxiv.org/abs/1506.02438)

4. **DPO**: Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (2023)

   - [Paper](https://arxiv.org/abs/2305.18290)

5. **IPO**: Azar et al. "A General Theoretical Paradigm to Understand Learning from Human Feedback" (2023)

   - [Paper](https://arxiv.org/abs/2310.12036)

6. **GRPO**: DeepSeek "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" (2025)

   - [Paper](https://arxiv.org/abs/2501.12948)

7. **RLOO**: Ahmadian et al. "Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs" (2024)
   - [Paper](https://arxiv.org/abs/2402.14740)

#### Distributed RL

8. **LlamaRL**: "A Distributed Asynchronous Reinforcement Learning Framework for Efficient Large-scale LLM Training" (2025)

   - [Paper](https://arxiv.org/abs/2505.24034)

9. **DistFlow**: "A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training" (2025)
   - [Paper](https://arxiv.org/abs/2507.13833)

#### Advanced Topics

10. **REINFORCE++**: "Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization" (2025)

    - [Paper](https://arxiv.org/abs/2501.03262)

11. **KL Regularization Design**: "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning" (2025)

    - [OpenReview](https://openreview.net/forum?id=qe060gmfm7)

12. **Token-Level RL**: "KL-Regularised Q-Learning: A Token-level Action-Value perspective on Online RLHF" (2025)
    - [Paper](https://arxiv.org/abs/2508.17000)

### Implementation Resources

1. **TRL (Transformers Reinforcement Learning)**

   - [Documentation](https://huggingface.co/docs/trl)
   - Implementations of PPO, DPO, GRPO, RLOO

2. **OpenRLHF**

   - [GitHub](https://github.com/OpenRLHF/OpenRLHF)
   - Distributed RLHF training framework

3. **DeepSpeed-Chat**

   - [GitHub](https://github.com/microsoft/DeepSpeedExamples)
   - Microsoft's RLHF training framework

4. **veRL**
   - [GitHub](https://github.com/volcengine/verl)
   - Volcano Engine's RL for LLMs

### Tutorials and Courses

1. **Hugging Face Deep RL Course**

   - [Course Link](https://huggingface.co/learn/deep-rl-course)
   - Great for RL fundamentals

2. **RLHF Book**

   - [Online Book](https://rlhfbook.com)
   - Comprehensive coverage of RLHF

3. **Spinning Up in Deep RL (OpenAI)**
   - [Documentation](https://spinningup.openai.com)
   - Core RL algorithm implementations

---

## Appendix A: Mathematical Derivations

### A.1 Policy Gradient Theorem Derivation

Starting from the objective:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$$

$$J(\theta) = \sum_\tau P(\tau | \theta) R(\tau)$$

Taking gradient:
$$\nabla_\theta J(\theta) = \sum_\tau \nabla_\theta P(\tau | \theta) R(\tau)$$

Using log-derivative trick: $\nabla_\theta P(\tau | \theta) = P(\tau | \theta) \nabla_\theta \log P(\tau | \theta)$

$$\nabla_\theta J(\theta) = \sum_\tau P(\tau | \theta) \nabla_\theta \log P(\tau | \theta) R(\tau) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log P(\tau | \theta) R(\tau)]$$

For a trajectory: $\log P(\tau | \theta) = \log P(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t | s_t) + \sum_{t=0}^{T} \log P(s_{t+1} | s_t, a_t)$

Only the policy term depends on $\theta$:
$$\nabla_\theta \log P(\tau | \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)$$

### A.2 Baseline Doesn't Introduce Bias

For any baseline $b(s)$ not depending on action:

$$\mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)] = b(s) \cdot \mathbb{E}_a[\nabla_\theta \log \pi_\theta(a|s)]$$

$$= b(s) \cdot \nabla_\theta \mathbb{E}_a[1] = b(s) \cdot \nabla_\theta 1 = 0$$

Therefore, subtracting $b(s)$ from returns leaves the gradient expectation unchanged.

### A.3 DPO Optimal Policy

Starting from the regularized RL objective:
$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta D_{\text{KL}}(\pi \| \pi_{\text{ref}})$$

The Lagrangian leads to:
$$\pi^*(y | x) = \frac{\pi_{\text{ref}}(y | x) \exp(r(x, y) / \beta)}{Z(x)}$$

Where $Z(x) = \sum_y \pi_{\text{ref}}(y | x) \exp(r(x, y) / \beta)$.

Verification: This satisfies the KKT conditions for the constrained optimization.

---

## Appendix B: Interview Preparation Checklist

### Conceptual Understanding

- [ ] Explain MDP formulation for LLMs (state, action, reward)
- [ ] Derive policy gradient theorem from scratch
- [ ] Explain why baselines don't introduce bias
- [ ] Articulate difference between V, Q, A, and reward model
- [ ] Explain token-level vs trajectory-level RL trade-offs

### Algorithm Deep Dives

- [ ] Walk through PPO clipping mechanism on whiteboard
- [ ] Derive DPO loss from RLHF objective
- [ ] Explain why GRPO doesn't need a critic
- [ ] Compare IPO vs DPO theoretically
- [ ] Explain GAE λ parameter's bias-variance trade-off

### Practical Knowledge

- [ ] Discuss distributed RL training challenges
- [ ] Explain KL penalty computation and tuning
- [ ] Discuss reward model training and limitations
- [ ] Compare on-policy vs off-policy for LLM training
- [ ] Discuss when to use which algorithm

### Recent Research Awareness

- [ ] DeepSeek-R1 training methodology
- [ ] REINFORCE++ and global normalization
- [ ] Process reward models
- [ ] Token-level Q-learning for LLMs

---

_Last updated: February 2026_
_Good luck with your studies and interview preparation!_
