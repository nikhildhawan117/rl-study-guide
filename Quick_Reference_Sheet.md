# RL for LLM Post-Training: Quick Reference Sheet

**One-page reference for interview preparation**

---

## Core Equations at a Glance

### Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]$$

### Value Functions

| Function     | Formula                                             | Meaning                                  |
| ------------ | --------------------------------------------------- | ---------------------------------------- |
| $V^\pi(s)$   | $\mathbb{E}[\sum_t \gamma^t r_t \mid s_0=s]$        | Expected return from state               |
| $Q^\pi(s,a)$ | $\mathbb{E}[\sum_t \gamma^t r_t \mid s_0=s, a_0=a]$ | Expected return after action             |
| $A^\pi(s,a)$ | $Q(s,a) - V(s)$                                     | Advantage (how much better than average) |

### PPO Clipped Objective

$$L^{\text{CLIP}} = \mathbb{E}\left[\min\left(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right], \quad r_t = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}$$

### GRPO Advantage

$$\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G} \quad \text{(normalize within group)}$$

### DPO Loss

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \beta \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right)$$

### KL Divergence (Monte Carlo)

$$\hat{D}_{\text{KL}} = \sum_t \left[\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_{\text{ref}}(y_t|x,y_{<t})\right]$$

### GAE

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### Bradley-Terry

$$P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$$

---

## Algorithm Decision Tree

```
Need to align LLM with preferences?
│
├─► Have only offline preference data?
│   ├─► Preferences are clean → DPO
│   └─► Preferences are noisy → IPO
│
├─► Can do online RL?
│   ├─► Have compute for critic?
│   │   └─► Yes → PPO (most stable)
│   │
│   ├─► Memory constrained?
│   │   ├─► Verifiable rewards → GRPO
│   │   └─► Need unbiased baseline → RLOO
│   │
│   └─► Want simplest setup?
│       └─► REINFORCE++
│
└─► Reasoning/code with execution feedback?
    └─► GRPO with rule-based rewards
```

---

## Key Distinctions

| Concept              | Definition                         | Example in LLM Context               |
| -------------------- | ---------------------------------- | ------------------------------------ |
| **State**            | Current context                    | Prompt + generated tokens so far     |
| **Action**           | Decision at each step              | Next token (or full response)        |
| **Reward Model**     | Scores outputs (fixed)             | Bradley-Terry trained on preferences |
| **Value Function**   | Predicts expected reward (learned) | Critic network in PPO                |
| **Reference Policy** | Frozen baseline                    | Initial SFT model                    |

---

## Variance Reduction Techniques

| Technique                | How It Works                  | Bias?                     |
| ------------------------ | ----------------------------- | ------------------------- |
| **Constant baseline**    | Subtract mean reward          | No                        |
| **Value baseline**       | Subtract $V(s)$               | Yes (if V is wrong)       |
| **Group normalization**  | Normalize within prompt group | Yes (small groups)        |
| **Global normalization** | Normalize across batch        | Vanishes with large batch |
| **Leave-one-out**        | Exclude self from baseline    | No                        |

---

## Hyperparameter Cheat Sheet

| Parameter             | Typical Value | Purpose                 |
| --------------------- | ------------- | ----------------------- |
| $\epsilon$ (PPO clip) | 0.1-0.2       | Trust region size       |
| $\beta$ (KL coeff)    | 0.01-0.1      | Regularization strength |
| $\lambda$ (GAE)       | 0.95          | Bias-variance tradeoff  |
| $\gamma$ (discount)   | 1.0           | Future reward weight    |
| Group size (GRPO)     | 8-16          | Samples per prompt      |
| Temperature           | 0.7-1.0       | Sampling diversity      |

---

## Common Interview Questions

1. **"Derive the policy gradient theorem"**

   - Use log-derivative trick: $\nabla P = P \nabla \log P$
   - Only policy terms depend on $\theta$

2. **"Why doesn't the baseline introduce bias?"**

   - $\mathbb{E}[\nabla \log \pi \cdot b] = b \cdot \nabla \mathbb{E}[1] = 0$

3. **"How does PPO clipping work?"**

   - Prevents ratio from going outside $[1-\epsilon, 1+\epsilon]$
   - Gradient becomes zero when clipping activates

4. **"Why doesn't GRPO need a critic?"**

   - Uses group mean as baseline instead of learned V(s)
   - Trade-off: requires multiple samples per prompt

5. **"How is KL computed in practice?"**

   - Monte Carlo: sum of log-prob differences over generated tokens
   - Reference model stays frozen

6. **"DPO vs RL-based methods?"**
   - DPO: offline, no sampling, simpler, but can't iterate
   - RL: online, can improve with more data, more complex

---

## Red Flags to Watch For

- **Reward hacking**: Policy finds adversarial high-reward outputs
- **Mode collapse**: Policy generates same output repeatedly
- **KL explosion**: Policy diverges too far from reference
- **Degenerate groups**: All samples in group have same reward (no gradient)
- **Credit assignment**: Which token caused the reward?

---

## Recent Trends (2024-2025)

1. **Critic-free methods** dominating (GRPO, RLOO, REINFORCE++)
2. **Verifiable rewards** for reasoning (code execution, math verification)
3. **Process reward models** for better credit assignment
4. **Distributed training** at 1000+ GPU scale
5. **Off-policy variants** for sample efficiency
6. **Token-level Q-learning** as alternative to policy gradient

---

_Print this for quick reference during interview prep!_
