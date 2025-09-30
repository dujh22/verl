# 近端策略优化 (PPO)

近端策略优化 (PPO) 是强化学习中策略梯度方法的一个系列，由 OpenAI 在 2017 年提出。PPO 在简洁性、稳定性和性能之间取得了平衡，使其成为现代强化学习应用中最广泛使用的算法之一，包括大规模语言模型的微调。

传统的策略梯度方法如 REINFORCE 或 Vanilla Policy Gradient 存在以下问题：

- 高方差和样本效率低下
- 由于大的策略更新导致的不稳定性

PPO 通过使用裁剪代理目标来解决这个问题，避免过大的更新而不需要二阶导数。

有关 PPO 的更多技术细节，我们建议阅读 [OpenAI spinning up 教程](https://spinningup.openai.com/en/latest/algorithms/ppo.html) 中的介绍，以及论文 [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)。

## 核心组件

- **Actor-Critic 架构**：PPO 需要同时具备 actor 模型（策略）和 critic 模型（价值函数）。这与其他算法如 GRPO 和 RLOO 不同，后者不需要 critic 模型。
- **广义优势估计 (GAE)**：PPO 使用 GAE 来计算优势值，这有助于减少策略梯度估计的方差，同时保持低偏差。
- **裁剪代理目标**：PPO 的核心通过裁剪代理目标函数实现，该函数限制策略更新。

## 配置

注意，所有包含 `micro_batch_size` 的配置用于配置每次前向或后向传播的最大样本或 token 数量，以避免 GPU 内存溢出，其值不应改变算法/收敛行为。

大多数 critic 配置与 actor 配置类似。注意下图省略了 critic 模型。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `data.train_batch_size`：用于生成一组采样轨迹/rollout 的提示的全局批次大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`
- `actor_rollout_ref.actor.ppo_mini_batch_size`：采样的轨迹集合被分割成多个小批次，批次大小为 ppo_mini_batch_size，用于 PPO actor 更新。ppo_mini_batch_size 是所有工作者的全局大小
- `actor_rollout_ref.critic.ppo_mini_batch_size`：采样的轨迹集合被分割成多个小批次，批次大小为 ppo_mini_batch_size，用于 PPO critic 更新。ppo_mini_batch_size 是所有工作者的全局大小
- `actor_rollout_ref.actor.clip_ratio`：PPO 裁剪范围。默认为 0.2
- `actor_rollout_ref.actor.ppo_epochs`：在一组采样轨迹上进行 PPO 更新的 epoch 数量（用于 actor）
- `critic.ppo_epochs`：在一组采样轨迹上进行 PPO 更新的 epoch 数量（用于 critic）。默认为 `actor_rollout_ref.actor.ppo_epochs`
- `algorithm.gamma`：折扣因子
- `algorithm.lam`：在 GAE 估计器中权衡偏差和方差的 lambda 项
- `algorithm.adv_estimator`：支持 gae、grpo、reinforce_plus_plus、reinforce_plus_plus_baseline、rloo、rloo_vectorized

## 高级扩展

### KL 散度控制

防止策略偏离参考策略太远的选项。提供两种机制：KL 奖励惩罚和 KL 损失。更多技术细节，请参见 [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

使用 KL 损失进行 KL 散度控制的选项：

- `actor_rollout_ref.actor.use_kl_loss`：在 actor 中使用 kl 损失。使用时，我们不在奖励函数中应用 KL。默认为 False
- `actor_rollout_ref.actor.kl_loss_coef`：kl 损失的系数。默认为 0.001
- `actor_rollout_ref.actor.kl_loss_type`：支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。在末尾添加 "+"（例如 'k1+' 和 'k3+'）将应用直通以使用 k2 进行无偏梯度估计，无论 kl 值估计如何（更多详细信息请参见 https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848）。如何计算 actor 和参考策略之间的 kl 散度。详细分析请参见此博客文章：http://joschu.net/blog/kl-approx.html

在奖励中使用 KL 惩罚的选项：

- `algorithm.use_kl_in_reward`：是否启用奖励内 kl 惩罚。默认为 False
- `algorithm.kl_penalty`：支持 kl(k1)、abs、mse(k2)、low_var_kl(k3) 和 full。这定义了计算 actor 和参考策略之间 kl 散度的方法。具体选项请参考 core_algos.py 中的 `kl_penalty`。详细分析请参见此博客文章：http://joschu.net/blog/kl-approx.html
- `algorithm.kl_ctrl.kl_coef`：奖励内 kl_penalty 的（初始）系数。默认为 0.001
- `algorithm.kl_ctrl.type`：'fixed' 用于 FixedKLController，'adaptive' 用于 AdaptiveKLController
- `algorithm.kl_ctrl.horizon`：详细信息请参见 AdaptiveKLController 的源代码
- `algorithm.kl_ctrl.target_kl`：详细信息请参见 AdaptiveKLController 的源代码

### 双裁剪 PPO

双裁剪 PPO 引入了一种方法，当优势小于零时，对策略比率应用下界，当乘以大比率时，不超过指定的下界。

![image](https://github.com/user-attachments/assets/fc232181-d8b0-4307-8dd2-4dc0a4c1c139)

- `actor_rollout_ref.actor.clip_ratio_c`：双裁剪 PPO 的值下界，默认为 3.0

## 参考示例

Qwen2.5 训练日志和命令：[链接](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log)

```bash
bash run_gemma.sh
  trainer.n_gpus_per_node=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  trainer.logger=console \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  data.train_batch_size=256 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size=2 \
  critic.ppo_micro_batch_size=2
```

verl v0.2 的参考性能：

| 模型                       | 方法       | 分数 | 链接                                                                                                                                  |
| -------------------------- | ---------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen/Qwen2.5-0.5B-Instruct | 预训练模型 | 36.4 | [Qwen 博客](https://qwenlm.github.io/blog/qwen2.5-llm/)                                                                                  |
| Qwen/Qwen2.5-0.5B-Instruct | PPO        | 56.7 | [PPO 命令和日志](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/Qwen2.5-0.5B-bsz256_2-prompt1024-resp512-0.567.log) |
