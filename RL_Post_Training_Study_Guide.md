# Comprehensive Reinforcement Learning Study Guide for LLM Post-Training

**Prepared for: Microsoft Superintelligence Post-Training Interview**  
**Background: Code Post-Training Research at DeepMind (RL & Evals for Gemini Code)**

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

The goal of policy gradient methods is to maximize expected reward:

$$J(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot|x)}[r(x, y)]$$

**The Policy Gradient Theorem** gives us:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R_t\right]$$

Where $R_t$ is the "reward-to-go" from time $t$ onwards.

**Intuition**:

- $\nabla_\theta \log \pi_\theta(a_t | s_t)$ points in the direction that increases probability of action $a_t$
- We weight this by how good that action was ($R_t$)
- Good actions get reinforced, bad actions get suppressed

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

**Your Question**: _Is the action the entire rollout or is it each token? Do we need to compute Q and V at each token?_

**Answer**: It depends on the algorithm, but here's the typical setup:

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

Where:

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the **probability ratio**
- $\epsilon$ is typically 0.2
- $A_t$ is the advantage estimate

#### How Clipping Works

```
If A_t > 0 (good action):
    - We want to increase π(a|s), so r_t > 1 is good
    - But clip(r_t, 0.8, 1.2) prevents r_t from exceeding 1.2
    - Gradient is zero once r_t > 1.2

If A_t < 0 (bad action):
    - We want to decrease π(a|s), so r_t < 1 is good
    - But clip prevents r_t from going below 0.8
    - Gradient is zero once r_t < 0.8
```

**Intuition**: Don't change any action's probability by more than ~20% per update.

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

**Intuition**: Advantage measures how much better than average this response is, normalized by the group's variance.

#### GRPO Objective

$$L^{\text{GRPO}}(\theta) = \mathbb{E}\left[\sum_{i=1}^{G} \min\left(r_t^{(i)}(\theta) \hat{A}_i, \text{clip}(r_t^{(i)}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i\right)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$

Where $r_t^{(i)}(\theta) = \frac{\pi_\theta(y_i | x)}{\pi_{\theta_{\text{old}}}(y_i | x)}$

### 6.3 KL Divergence Penalty

**Your Question**: _Is it always the base model or does it update? How is it computed?_

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

This is the **reverse KL** estimate using only the generated tokens.

#### Implementation

```python
def compute_kl_penalty(policy_logprobs, ref_logprobs):
    """
    policy_logprobs: log π_θ(y_t | context) for each token
    ref_logprobs: log π_ref(y_t | context) for each token
    """
    kl = policy_logprobs - ref_logprobs  # Per-token KL
    return kl.sum()  # Sum over sequence
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
_Good luck with your interview!_
