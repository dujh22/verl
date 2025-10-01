# 组相对策略优化 (GRPO)

在强化学习中，PPO等经典算法依赖于"评判者"(critic)模型来估计动作的价值，指导学习过程。然而，训练这个评判者模型可能非常耗费资源。

GRPO通过消除对单独评判者模型的需求来简化这个过程。它的工作原理如下：
- 组采样：对于给定问题，模型生成多个可能的解决方案，形成输出的"组"。
- 奖励分配：每个解决方案都会被评估并根据其正确性或质量分配奖励。
- 基线计算：组的平均奖励作为基线。
- 策略更新：模型通过将每个解决方案的奖励与组基线进行比较来更新其参数，强化优于平均水平的解决方案，抑制低于平均水平的解决方案。

这种方法通过避免训练单独的价值估计模型来减少计算开销，使学习过程更加高效。更多详细信息，请参考原论文 [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/pdf/2402.03300)

## 核心组件

- 无价值函数（无评判者）：与PPO不同，GRPO不训练单独的价值网络（评判者）
- 组采样（分组滚出）：GRPO不是每个输入评估一个滚出，而是为每个提示从当前策略生成多个完成（响应）。这组完成被称为一个组。
- 相对奖励：在每个组内，完成会被评分（例如基于正确性），奖励相对于组进行标准化。

## 配置

注意，所有包含 `micro_batch_size` 的配置用于配置每次前向或后向传递的最大样本或令牌数量，以避免GPU内存不足，其值不应改变算法/收敛行为。

尽管许多配置以 `ppo_` 前缀开头，但它们在verl的不同RL算法中都有效，因为GRPO训练循环与PPO（无评判者）类似。

![image](https://github.com/user-attachments/assets/16aebad1-0da6-4eb3-806d-54a74e712c2d)

- `actor_rollout.ref.rollout.n`：对于每个提示，采样n次。默认为1。对于GRPO，请将其设置为大于1的值以进行组采样。

- `data.train_batch_size`：用于生成一组采样轨迹/滚出的提示的全局批量大小。响应/轨迹的数量为 `data.train_batch_size * actor_rollout.ref.rollout.n`

- `actor_rollout_ref.actor.ppo_mini_batch_size`：采样的轨迹集被分割成多个小批量，批量大小为ppo_mini_batch_size，用于PPO参与者更新。ppo_mini_batch_size是所有工作者的全局大小。

- `actor_rollout_ref.actor.ppo_epochs`：在参与者的一组采样轨迹上进行GRPO更新的轮数

- `actor_rollout_ref.actor.clip_ratio`：GRPO裁剪范围。默认为0.2

- `algorithm.adv_estimator`：默认为gae。请将其设置为grpo

- `actor_rollout_ref.actor.loss_agg_mode`：默认为"token-mean"。选项包括"token-mean"、"seq-mean-token-sum"、"seq-mean-token-mean"。原始GRPO论文采用样本级损失（seq-mean-token-mean），在长链思维场景中可能不稳定。verl提供的所有GRPO示例脚本使用默认配置"token-mean"进行损失聚合。

GRPO不是通过在奖励中添加KL惩罚来正则化，而是通过直接在损失中添加训练策略和参考策略之间的KL散度来正则化：

- `actor_rollout_ref.actor.use_kl_loss`：在参与者中使用kl损失。使用时，我们不在奖励函数中应用KL。默认为False。对于GRPO，请设置为True。

- `actor_rollout_ref.actor.kl_loss_coef`：kl损失的系数。默认为0.001。

- `actor_rollout_ref.actor.kl_loss_type`：支持kl(k1)、abs、mse(k2)、low_var_kl(k3)和full。在末尾添加"+"（例如'k1+'和'k3+'）将应用直通以使用k2进行无偏梯度估计，无论kl值估计如何（详见 https://github.com/volcengine/verl/pull/2953#issuecomment-3162113848）。如何计算参与者和参考策略之间的kl散度。详见此博客文章分析：http://joschu.net/blog/kl-approx.html

## 高级扩展

### DrGRPO

研究 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/pdf/2503.20783) 声称GRPO存在优化偏差，导致人工产生更长的响应，特别是对于不正确的输出。这种低效率源于GRPO使用基于组的奖励标准化计算优势的方式，可能无意中偏向更长、不太准确的响应。相反，DrGRPO通过用全局常数标准化来聚合令牌级损失，消除长度偏差。

要启用DrGRPO，请配置以下内容，其他参数与GRPO相同：

- `actor_rollout_ref.actor.loss_agg_mode`："seq-mean-token-sum-norm"，关闭序列维度平均
- `actor_rollout_ref.actor.use_kl_loss`：对于DrGRPO，请设置为False
- `algorithm.norm_adv_by_std_in_grpo`：False，关闭标准差标准化

## 参考示例

Qwen2.5 GRPO训练日志和命令：[链接](https://github.com/eric-haibin-lin/verl-data/blob/experiments/gsm8k/qwen2-7b-fsdp2.log)

```bash
bash examples/grpo_trainer/run_qwen3-8b.sh
```

更多参考性能，请参见 https://verl.readthedocs.io/en/latest/algo/baseline.html
