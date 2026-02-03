"""
Simplified Code Examples for LLM RL Algorithms
===============================================

These are pedagogical implementations - not production code.
They illustrate the core ideas without the complexity of real systems.

Run with: python Code_Examples.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

# ============================================================================
# SECTION 1: REWARD MODEL TRAINING (Bradley-Terry)
# ============================================================================


def bradley_terry_loss(
    reward_model: nn.Module,
    prompts: torch.Tensor,
    chosen_responses: torch.Tensor,
    rejected_responses: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry loss for reward model training.

    Given preference pairs (y_w preferred over y_l), train reward model
    to satisfy: r(y_w) > r(y_l)

    Loss = -log(sigmoid(r(y_w) - r(y_l)))
    """
    # Get rewards for chosen and rejected
    r_chosen = reward_model(prompts, chosen_responses)  # Shape: (batch,)
    r_rejected = reward_model(prompts, rejected_responses)  # Shape: (batch,)

    # Bradley-Terry loss
    logits = r_chosen - r_rejected
    loss = -F.logsigmoid(logits).mean()

    return loss


# ============================================================================
# SECTION 2: REINFORCE WITH BASELINE
# ============================================================================


def reinforce_loss(
    policy: nn.Module,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float = None,
) -> torch.Tensor:
    """
    Basic REINFORCE policy gradient.

    Loss = -E[log π(y|x) * (R - baseline)]
    """
    # Get log probabilities of each token in response
    # Shape: (batch, seq_len)
    log_probs = get_sequence_log_probs(policy, prompts, responses)

    # Sum log probs over sequence for trajectory-level
    # Shape: (batch,)
    trajectory_log_probs = log_probs.sum(dim=-1)

    # Compute advantage
    if baseline is None:
        baseline = rewards.mean()  # Simple mean baseline
    advantages = rewards - baseline

    # REINFORCE loss (negative because we maximize)
    loss = -(trajectory_log_probs * advantages).mean()

    return loss


def rloo_loss(
    policy: nn.Module,
    prompts: torch.Tensor,  # (num_prompts,)
    responses: torch.Tensor,  # (num_prompts, K, seq_len) - K samples per prompt
    rewards: torch.Tensor,  # (num_prompts, K)
) -> torch.Tensor:
    """
    REINFORCE Leave-One-Out loss.

    For K samples per prompt, the baseline for sample i excludes sample i:
    b_i = (1/(K-1)) * sum_{j != i} R_j

    This gives an unbiased baseline.
    """
    num_prompts, K, seq_len = responses.shape

    total_loss = 0.0

    for p in range(num_prompts):
        prompt = prompts[p]
        prompt_responses = responses[p]  # (K, seq_len)
        prompt_rewards = rewards[p]  # (K,)

        for i in range(K):
            # Leave-one-out baseline: mean of all rewards except i
            mask = torch.ones(K, dtype=torch.bool)
            mask[i] = False
            baseline = prompt_rewards[mask].mean()

            # Advantage for sample i
            advantage = prompt_rewards[i] - baseline

            # Log probability of response i
            log_prob = get_sequence_log_probs(
                policy, prompt.unsqueeze(0), prompt_responses[i].unsqueeze(0)
            )
            trajectory_log_prob = log_prob.sum()

            # Accumulate loss
            total_loss -= trajectory_log_prob * advantage

    return total_loss / (num_prompts * K)


# ============================================================================
# SECTION 3: PPO
# ============================================================================


def ppo_loss(
    policy: nn.Module,
    old_policy: nn.Module,  # Frozen copy from before update
    value_net: nn.Module,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    old_log_probs: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,  # GAE lambda
    epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Proximal Policy Optimization loss with GAE.

    Includes:
    - Clipped policy objective
    - Value function loss
    - Entropy bonus
    """
    batch_size, seq_len = responses.shape

    # Get current log probs and values
    # Shape: (batch, seq_len)
    log_probs = get_sequence_log_probs(policy, prompts, responses)
    values = value_net(prompts, responses)  # (batch, seq_len)

    # Compute advantages using GAE
    # For simplicity, assume reward only at last token
    advantages = compute_gae(values, rewards, gamma, lam)  # (batch, seq_len)

    # Normalize advantages (common practice)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss with clipping
    ratio = torch.exp(log_probs - old_log_probs)  # π_new / π_old

    # Clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

    # Value loss
    returns = advantages + values.detach()  # GAE targets
    value_loss = F.mse_loss(values, returns)

    # Entropy bonus (encourages exploration)
    entropy = compute_entropy(policy, prompts, responses)

    # Combined loss
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    metrics = {
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy.item(),
        "mean_ratio": ratio.mean().item(),
    }

    return total_loss, metrics


def compute_gae(
    values: torch.Tensor,  # (batch, seq_len)
    rewards: torch.Tensor,  # (batch,) - terminal reward
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """
    Generalized Advantage Estimation.

    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)

    For LLM with terminal reward, r_t = 0 for t < T, r_T = reward
    """
    batch_size, seq_len = values.shape
    advantages = torch.zeros_like(values)

    # Start from the end and work backwards
    last_gae = 0

    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # Last timestep: use terminal reward
            next_value = 0  # Terminal state
            delta = rewards + gamma * next_value - values[:, t]
        else:
            # Intermediate timesteps: r_t = 0
            next_value = values[:, t + 1]
            delta = 0 + gamma * next_value - values[:, t]

        # GAE formula
        advantages[:, t] = delta + gamma * lam * last_gae
        last_gae = advantages[:, t]

    return advantages


# ============================================================================
# SECTION 4: GRPO
# ============================================================================


def grpo_loss(
    policy: nn.Module,
    ref_policy: nn.Module,  # Frozen reference model
    prompts: torch.Tensor,  # (num_prompts,)
    responses: torch.Tensor,  # (num_prompts, G, seq_len) - G samples per prompt
    rewards: torch.Tensor,  # (num_prompts, G)
    old_log_probs: torch.Tensor,  # (num_prompts, G, seq_len)
    epsilon: float = 0.2,
    beta: float = 0.1,  # KL coefficient
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Group Relative Policy Optimization loss.

    Key differences from PPO:
    1. No value function - uses group mean as baseline
    2. KL penalty in loss, not reward
    3. Advantage normalized within each group
    """
    num_prompts, G, seq_len = responses.shape

    total_loss = 0.0
    total_kl = 0.0

    for p in range(num_prompts):
        prompt = prompts[p]
        group_responses = responses[p]  # (G, seq_len)
        group_rewards = rewards[p]  # (G,)
        group_old_log_probs = old_log_probs[p]  # (G, seq_len)

        # Compute group-relative advantages
        # A_i = (r_i - mean) / std
        mean_reward = group_rewards.mean()
        std_reward = group_rewards.std() + 1e-8
        advantages = (group_rewards - mean_reward) / std_reward  # (G,)

        for i in range(G):
            response = group_responses[i].unsqueeze(0)  # (1, seq_len)
            old_lp = group_old_log_probs[i].unsqueeze(0)  # (1, seq_len)
            advantage = advantages[i]

            # Current log probs
            log_probs = get_sequence_log_probs(policy, prompt.unsqueeze(0), response)

            # Reference log probs for KL
            with torch.no_grad():
                ref_log_probs = get_sequence_log_probs(
                    ref_policy, prompt.unsqueeze(0), response
                )

            # Probability ratio for clipping
            ratio = torch.exp(log_probs.sum() - old_lp.sum())
            clipped_ratio = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)

            # Clipped policy loss
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            # KL penalty (per token, summed)
            kl = (log_probs - ref_log_probs).sum()
            total_kl += kl

            # Combined loss
            total_loss += policy_loss + beta * kl

    num_samples = num_prompts * G

    metrics = {
        "policy_loss": (total_loss / num_samples).item(),
        "mean_kl": (total_kl / num_samples).item(),
    }

    return total_loss / num_samples, metrics


# ============================================================================
# SECTION 5: DPO
# ============================================================================


def dpo_loss(
    policy: nn.Module,
    ref_policy: nn.Module,  # Frozen reference model
    prompts: torch.Tensor,
    chosen_responses: torch.Tensor,
    rejected_responses: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Direct Preference Optimization loss.

    L_DPO = -log σ(β * (log π/π_ref for chosen - log π/π_ref for rejected))

    No RL, no reward model - just supervised learning on preferences.
    """
    # Log probs from policy
    policy_chosen_lp = get_sequence_log_probs(policy, prompts, chosen_responses).sum(
        dim=-1
    )
    policy_rejected_lp = get_sequence_log_probs(
        policy, prompts, rejected_responses
    ).sum(dim=-1)

    # Log probs from reference (frozen)
    with torch.no_grad():
        ref_chosen_lp = get_sequence_log_probs(
            ref_policy, prompts, chosen_responses
        ).sum(dim=-1)
        ref_rejected_lp = get_sequence_log_probs(
            ref_policy, prompts, rejected_responses
        ).sum(dim=-1)

    # Implicit rewards: r = β * log(π/π_ref)
    chosen_rewards = beta * (policy_chosen_lp - ref_chosen_lp)
    rejected_rewards = beta * (policy_rejected_lp - ref_rejected_lp)

    # DPO loss
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()

    return loss


# ============================================================================
# SECTION 6: KL DIVERGENCE COMPUTATION
# ============================================================================


def compute_kl_divergence(
    policy: nn.Module,
    ref_policy: nn.Module,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    method: str = "monte_carlo",
) -> torch.Tensor:
    """
    Compute KL divergence between policy and reference.

    Two methods:
    1. Monte Carlo: Use generated tokens as samples
    2. Full: Sum over entire vocabulary (expensive)
    """
    if method == "monte_carlo":
        # Most common: just use the generated tokens
        policy_lp = get_sequence_log_probs(policy, prompts, responses)
        with torch.no_grad():
            ref_lp = get_sequence_log_probs(ref_policy, prompts, responses)

        # KL ≈ log π_θ(y) - log π_ref(y) for sampled y
        kl_per_token = policy_lp - ref_lp
        kl_per_sequence = kl_per_token.sum(dim=-1)

        return kl_per_sequence.mean()

    elif method == "full":
        # Full KL: sum over vocabulary (expensive, rarely used)
        batch_size, seq_len = responses.shape
        total_kl = 0.0

        for t in range(seq_len):
            context = responses[:, :t]  # Everything before t

            # Get distributions
            policy_dist = get_token_distribution(policy, prompts, context)
            ref_dist = get_token_distribution(ref_policy, prompts, context)

            # KL(π || π_ref) = Σ π(v) log(π(v)/π_ref(v))
            kl = (policy_dist * (policy_dist.log() - ref_dist.log())).sum(dim=-1)
            total_kl += kl.mean()

        return total_kl

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# SECTION 7: COMPLETE TRAINING LOOP EXAMPLES
# ============================================================================


def train_grpo_step(
    policy: nn.Module,
    ref_policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: List[str],
    tokenizer,
    reward_fn,  # Can be reward model or code verifier
    group_size: int = 8,
    temperature: float = 1.0,
    beta: float = 0.1,
):
    """
    One GRPO training step.

    1. Sample multiple responses per prompt
    2. Score with reward function
    3. Compute group-relative advantages
    4. Update policy with clipped objective + KL
    """
    policy.train()

    all_responses = []
    all_rewards = []
    all_old_log_probs = []

    # Phase 1: Generate responses
    with torch.no_grad():
        for prompt in prompts:
            prompt_responses = []
            prompt_rewards = []
            prompt_log_probs = []

            for _ in range(group_size):
                # Sample response
                response, log_probs = sample_response(
                    policy, prompt, tokenizer, temperature
                )

                # Score response
                reward = reward_fn(prompt, response)

                prompt_responses.append(response)
                prompt_rewards.append(reward)
                prompt_log_probs.append(log_probs)

            all_responses.append(prompt_responses)
            all_rewards.append(torch.tensor(prompt_rewards))
            all_old_log_probs.append(torch.stack(prompt_log_probs))

    # Phase 2: Compute loss and update
    optimizer.zero_grad()

    total_loss = 0.0
    total_kl = 0.0
    num_samples = 0

    for p, prompt in enumerate(prompts):
        group_rewards = all_rewards[p]

        # Group-relative advantage
        mean_r = group_rewards.mean()
        std_r = group_rewards.std() + 1e-8
        advantages = (group_rewards - mean_r) / std_r

        for i in range(group_size):
            response = all_responses[p][i]
            old_lp = all_old_log_probs[p][i]
            advantage = advantages[i]

            # Current log probs
            new_lp = get_response_log_probs(policy, prompt, response, tokenizer)

            # Reference log probs
            with torch.no_grad():
                ref_lp = get_response_log_probs(ref_policy, prompt, response, tokenizer)

            # Ratio for clipping
            ratio = torch.exp(new_lp.sum() - old_lp.sum())
            clipped_ratio = torch.clamp(ratio, 0.8, 1.2)

            # Policy loss (clipped)
            policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)

            # KL penalty
            kl = (new_lp - ref_lp).sum()

            total_loss += policy_loss + beta * kl
            total_kl += kl.item()
            num_samples += 1

    # Backward and update
    mean_loss = total_loss / num_samples
    mean_loss.backward()
    optimizer.step()

    return {
        "loss": mean_loss.item(),
        "mean_kl": total_kl / num_samples,
        "mean_reward": torch.cat(all_rewards).mean().item(),
    }


def train_ppo_step(
    policy: nn.Module,
    value_net: nn.Module,
    ref_policy: nn.Module,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    prompts: List[str],
    tokenizer,
    reward_fn,
    num_epochs: int = 4,
    minibatch_size: int = 8,
):
    """
    One PPO training step (multiple epochs on same batch).

    1. Collect rollouts
    2. Compute advantages (GAE)
    3. Multiple epochs of updates on same data
    """
    # Phase 1: Collect rollouts
    rollouts = []

    with torch.no_grad():
        for prompt in prompts:
            response, log_probs = sample_response(policy, prompt, tokenizer)
            values = get_values(value_net, prompt, response, tokenizer)
            reward = reward_fn(prompt, response)

            rollouts.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "log_probs": log_probs,
                    "values": values,
                    "reward": reward,
                }
            )

    # Phase 2: Compute GAE advantages
    for rollout in rollouts:
        rollout["advantages"] = compute_gae_for_rollout(
            rollout["values"], rollout["reward"]
        )
        rollout["returns"] = rollout["advantages"] + rollout["values"]

    # Phase 3: Multiple epochs
    for epoch in range(num_epochs):
        # Shuffle and create minibatches
        np.random.shuffle(rollouts)

        for i in range(0, len(rollouts), minibatch_size):
            batch = rollouts[i : i + minibatch_size]

            # Policy update
            policy_optimizer.zero_grad()
            policy_loss = compute_ppo_policy_loss(policy, batch)
            policy_loss.backward()
            policy_optimizer.step()

            # Value update
            value_optimizer.zero_grad()
            value_loss = compute_value_loss(value_net, batch)
            value_loss.backward()
            value_optimizer.step()

    return {
        "mean_reward": np.mean([r["reward"] for r in rollouts]),
    }


# ============================================================================
# HELPER FUNCTIONS (Stubs - would be implemented with actual model)
# ============================================================================


def get_sequence_log_probs(policy, prompts, responses):
    """Get log probability of each token in response."""
    # This would actually run the model and extract log probs
    # Stub returns random values for illustration
    batch_size = prompts.shape[0] if len(prompts.shape) > 0 else 1
    seq_len = responses.shape[-1]
    return torch.randn(batch_size, seq_len)


def get_token_distribution(policy, prompts, context):
    """Get probability distribution over vocabulary."""
    # Stub
    vocab_size = 50000
    batch_size = prompts.shape[0]
    return F.softmax(torch.randn(batch_size, vocab_size), dim=-1)


def compute_entropy(policy, prompts, responses):
    """Compute entropy of policy distribution."""
    # Stub
    return torch.tensor(1.0)


def sample_response(policy, prompt, tokenizer, temperature=1.0):
    """Sample a response from the policy."""
    # Stub
    return "Generated response", torch.randn(100)


def get_response_log_probs(policy, prompt, response, tokenizer):
    """Get log probs for a response."""
    # Stub
    return torch.randn(100)


def get_values(value_net, prompt, response, tokenizer):
    """Get value estimates for each position."""
    # Stub
    return torch.randn(100)


def compute_gae_for_rollout(values, reward):
    """Compute GAE for a single rollout."""
    # Stub - would implement the full GAE computation
    return torch.randn_like(values)


def compute_ppo_policy_loss(policy, batch):
    """Compute PPO clipped policy loss for a batch."""
    # Stub
    return torch.tensor(1.0, requires_grad=True)


def compute_value_loss(value_net, batch):
    """Compute value function MSE loss."""
    # Stub
    return torch.tensor(1.0, requires_grad=True)


# ============================================================================
# MAIN: Quick demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LLM RL Algorithm Code Examples")
    print("=" * 60)
    print()
    print("This file contains pedagogical implementations of:")
    print("  1. Bradley-Terry reward model training")
    print("  2. REINFORCE with baseline")
    print("  3. RLOO (REINFORCE Leave-One-Out)")
    print("  4. PPO with GAE")
    print("  5. GRPO (Group Relative Policy Optimization)")
    print("  6. DPO (Direct Preference Optimization)")
    print("  7. KL divergence computation")
    print()
    print("See the functions and their docstrings for details.")
    print("These are simplified for understanding - not production code.")
    print()
    print("Key equations implemented:")
    print()
    print("REINFORCE:    ∇J = E[∇log π(a|s) · (R - b)]")
    print("PPO:          L = min(r·A, clip(r)·A), r = π_new/π_old")
    print("GRPO:         A = (r - μ_group) / σ_group")
    print("DPO:          L = -log σ(β·(log π/π_ref)_w - β·(log π/π_ref)_l)")
    print("GAE:          A_t = Σ (γλ)^l δ_{t+l}")
    print()
