# Deep Dive: Frequently Confused Concepts in LLM RL

This document provides extended answers to specific questions that often cause confusion.

---

## 1. What Exactly is Q? Ground Truth or Learned?

This is a common source of confusion. Let's be precise:

### Q is NOT Ground Truth

The Q-function $Q^\pi(s, a)$ is **not** the actual reward from the environment. It is an **estimate** of the **expected cumulative future reward** if the agent takes action $a$ in state $s$ and then follows policy $\pi$.

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{T} \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$

### How Q Relates to Different Concepts

| Concept                           | What It Is                                                | Ground Truth?       |
| --------------------------------- | --------------------------------------------------------- | ------------------- |
| **Reward $r(s, a)$ or $r(x, y)$** | Actual signal from environment (reward model or verifier) | Yes                 |
| **Return $R(\tau) = \sum_t r_t$** | Sum of actual rewards in a trajectory                     | Yes (after rollout) |
| **Q-function $Q(s, a)$**          | _Prediction_ of expected return                           | No (learned)        |
| **Value function $V(s)$**         | _Prediction_ of expected return from state                | No (learned)        |

### The Relationship

After completing a rollout, the **actual return** $R$ is observed. The Q-function is trained to **predict** this return before the rollout has finished.

Think of it like:

- **Q(s, a)**: "If I take this action, I expect to get reward X"
- **R**: "I took the action and actually got reward Y"
- **Training Q**: Minimize $(Q(s,a) - R)^2$

### In LLM RLHF Specifically

In most LLM RLHF setups:

1. **Reward Model** $r_\phi(x, y)$: Trained on human preferences, provides the "ground truth" signal
2. **Value Function** $V_\psi(s)$: Trained during RL to predict expected reward
3. **Q-function**: Often not explicitly used; replaced by trajectory-level rewards + GAE

When **code execution feedback** is used (e.g., in code post-training), the reward is truly ground truth—the code either passes tests or doesn't. This is a **verifiable reward** and is simpler than learned reward models.

---

## 2. How is the Value Function Trained and Updated During RL?

### Training the Value Function

The value function (critic) is trained alongside the policy (actor) in algorithms like PPO. Here's the process:

#### Step 1: Collect Rollouts

```python
trajectories = []
for prompt in batch:
    response = policy.generate(prompt)  # Sample from π_θ
    reward = get_reward(prompt, response)  # From RM or verifier
    trajectories.append((prompt, response, reward))
```

#### Step 2: Compute Value Targets

For each state (token position) in the trajectory, compute the **target value**:

**Option A: Monte Carlo Return**
$$V_{\text{target}}(s_t) = \sum_{k=t}^{T} \gamma^{k-t} r_k$$

This is the actual return from time $t$ to the end.

**Option B: TD Target (with bootstrapping)**
$$V_{\text{target}}(s_t) = r_t + \gamma V_{\psi_{\text{old}}}(s_{t+1})$$

Uses current value estimate for future.

**Option C: GAE-based Target (most common)**
$$V_{\text{target}}(s_t) = V_\psi(s_t) + \hat{A}_t^{\text{GAE}}$$

Where $\hat{A}_t^{\text{GAE}}$ is the advantage estimate.

#### Step 3: Update Value Function

Minimize MSE between predictions and targets:

$$L_V(\psi) = \frac{1}{N}\sum_{t} \left(V_\psi(s_t) - V_{\text{target}}(s_t)\right)^2$$

```python
def update_value_function(value_net, states, returns, optimizer):
    predictions = value_net(states)
    loss = F.mse_loss(predictions, returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### When is V Updated?

In PPO, the value function is updated **every training iteration**, typically with multiple epochs on the same batch:

```python
for epoch in range(num_epochs):  # e.g., 4 epochs
    for minibatch in batch.split(minibatch_size):
        # Update policy
        policy_loss = compute_ppo_loss(policy, minibatch)
        policy_optimizer.step()

        # Update value function
        value_loss = compute_value_loss(value_net, minibatch)
        value_optimizer.step()
```

### Value Function Architecture

In LLM RLHF, the value function often:

- **Shares the backbone** with the policy (efficient but coupled)
- **Has a separate head** that outputs a scalar for each token position
- Or is a **completely separate model** (more memory, but decoupled training)

```python
class PolicyWithCritic(nn.Module):
    def __init__(self, base_model):
        self.backbone = base_model
        self.lm_head = nn.Linear(hidden_size, vocab_size)  # Policy
        self.value_head = nn.Linear(hidden_size, 1)        # Critic

    def forward(self, x):
        hidden = self.backbone(x)
        logits = self.lm_head(hidden)       # π(a|s)
        values = self.value_head(hidden)    # V(s)
        return logits, values
```

### Why GRPO/RLOO Skip the Value Function

Training a value function that's as large as the policy is expensive:

- **Memory**: Need to store optimizer states for value network
- **Compute**: Need forward and backward passes through value network
- **Instability**: Value function can be slow to converge

GRPO insight: Instead of learning V(s), use **empirical mean of the group** as baseline:

$$\hat{A}_i = r_i - \frac{1}{G}\sum_{j=1}^G r_j$$

This requires no learning, just multiple samples per prompt.

---

## 3. Rollouts and Advantage Computation: When Does What Happen?

### Common Question: "We usually allow the model to finish its rollout before computing advantage, right?"

**Yes, that's correct for most LLM RLHF setups.**

### Timeline of a Training Step

```
Time ────────────────────────────────────────────────────────►

┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: GENERATION (Rollout)                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  prompt → [generate tokens one by one] → complete response  │
│                                                              │
│  At each step:                                               │
│  - Sample token from π_θ(·|context)                          │
│  - Store log-prob for later                                  │
│  - (If using critic) Store V(s) for later                   │
│                                                              │
│  Result: Full trajectory τ = (y_1, y_2, ..., y_T)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: REWARD SCORING                                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Complete response → Reward Model → r(x, y)                  │
│                                                              │
│  For code: Execute → Check tests → Binary reward             │
│  For reasoning: Verify answer → Binary reward                │
│  For general: Learned reward model → Scalar reward           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: ADVANTAGE COMPUTATION                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Option A: Trajectory-level (GRPO, RLOO)                     │
│  ─────────────────────────────────────                       │
│  Single advantage for entire response:                       │
│  Â = (r - mean(group)) / std(group)                          │
│                                                              │
│  Option B: Token-level with GAE (PPO)                        │
│  ────────────────────────────────────                        │
│  For each token position t:                                  │
│  δ_t = r_t + γV(s_{t+1}) - V(s_t)                           │
│  Â_t = Σ (γλ)^l δ_{t+l}                                      │
│                                                              │
│  Note: r_t = 0 for t < T, r_T = final reward                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: POLICY UPDATE                                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Compute gradient: ∇_θ Σ_t log π_θ(y_t|s_t) · Â_t            │
│  Apply PPO clipping / trust region                           │
│  Update θ                                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Is the Action the Entire Rollout or Each Token?

**It depends on how the MDP is set up:**

#### Trajectory-Level Actions (Most Common in LLM RLHF)

- **Action**: Entire response $y = (y_1, \ldots, y_T)$
- **One advantage per trajectory**: $\hat{A}(y)$
- **Policy gradient**:

$$\nabla_\theta J = \mathbb{E}\left[\left(\sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t|s_t)\right) \cdot \hat{A}(y)\right]$$

The gradient accumulates over all tokens, weighted by a single advantage.

#### Token-Level Actions (More Sophisticated)

- **Action**: Each token $y_t$
- **One advantage per token**: $\hat{A}_t$
- **Policy gradient**:

$$\nabla_\theta J = \mathbb{E}\left[\sum_{t=1}^T \nabla_\theta \log \pi_\theta(y_t|s_t) \cdot \hat{A}_t\right]$$

Each token gets its own advantage, requiring Q/V at each position.

### Do We Need Q and V at Each Token?

**For trajectory-level**: No. Only the final reward is needed.

**For token-level with GAE**: Yes, but only V (not Q). The computation is:

- $V(s_t)$ at each token position
- Use TD errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
- Compute GAE advantages $\hat{A}_t$

**Practical note**: Even with sparse terminal reward ($r_t = 0$ for $t < T$), GAE propagates reward credit backward through the value function.

### Why Trajectory-Level is Popular

1. **Simpler**: No need for per-token value estimation
2. **Reward model fit**: Most reward models score complete responses
3. **Less noise**: Value function errors don't accumulate
4. **Cheaper**: No critic forward passes during generation

---

## 4. Code Execution Feedback: A Special Case

For code post-training applications, the setup simplifies as follows:

### Verifiable Rewards

For code tasks with test suites:

```python
def get_code_reward(prompt, code_response):
    try:
        exec(code_response)
        result = run_tests(code_response, test_cases)
        return 1.0 if result.all_pass else 0.0
    except:
        return 0.0  # Syntax error or runtime error
```

**Key difference from learned reward models**:

- This reward is **ground truth** (code either works or doesn't)
- No reward model training needed
- No reward hacking (can't fool the test suite)
- No distribution shift concerns

### Why GRPO Works Well for Code/Reasoning

1. **Binary rewards** are common (pass/fail)
2. **Verifiable**: Ground truth available
3. **No reward model needed**: Rule-based rewards suffice
4. **Group normalization**: Works well with binary outcomes

DeepSeek-R1's success demonstrates this: simple accuracy + format rewards, no learned reward model.

---

## 5. Summary: Clearing Up the Confusion

| Question                            | Answer                                                   |
| ----------------------------------- | -------------------------------------------------------- |
| Is Q ground truth?                  | No, Q is a learned prediction of expected return         |
| What's ground truth?                | The actual reward $r$ from RM or verifier                |
| When is V updated?                  | Every training iteration, alongside policy               |
| Do we wait for rollout to complete? | Yes, for most LLM RLHF setups                            |
| Is action = token or trajectory?    | Usually trajectory, but token-level exists               |
| Need Q/V at each token?             | Only if using token-level advantages (e.g., GAE)         |
| For code tasks specifically?        | Reward is ground truth (test execution), GRPO works well |

---

## 6. Mental Model for LLM RL

Think of LLM RL training as:

```
┌──────────────────────────────────────────────────────────────┐
│                     THE RL TRAINING LOOP                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. GENERATE: Let the model write complete responses          │
│     (collect trajectories)                                    │
│                                                               │
│  2. EVALUATE: Score each response                             │
│     - Reward model: learned from preferences                  │
│     - Code verifier: run tests, check correctness             │
│     - Rule-based: format checks, length penalties             │
│                                                               │
│  3. COMPARE: Figure out which responses are relatively good   │
│     - PPO: Use value function baseline                        │
│     - GRPO: Use group mean baseline                           │
│     - RLOO: Use leave-one-out baseline                        │
│                                                               │
│  4. REINFORCE: Increase probability of good responses         │
│     - Policy gradient: ∇ log π(y) · Advantage                 │
│     - Clipping: Don't change too much at once                 │
│     - KL penalty: Stay close to reference model               │
│                                                               │
│  5. REPEAT: Sample new prompts, generate new responses        │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

The key insight: **We're doing supervised learning where the "labels" (which actions to reinforce) come from RL's compare-and-reinforce mechanism rather than human annotation.**

---

_This document should resolve the most common sources of confusion when studying LLM RL._
