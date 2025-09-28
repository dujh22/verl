<div align="center">
 👋 大家好！ 
    verl 是由 <b>字节跳动 Seed 团队</b> 发起并由 verl 社区维护的强化学习训练库。
    <br>
    <br>
</div>

<div align="center">

`<a href="https://deepwiki.com/volcengine/verl"><img src="https://devin.ai/assets/deepwiki-badge.png" alt="Ask DeepWiki.com" style="height:20px;">``</a>`
[![GitHub Repo stars](https://img.shields.io/github/stars/volcengine/verl)](https://github.com/volcengine/verl/stargazers)
[![Twitter](https://img.shields.io/twitter/follow/verl_project)](https://twitter.com/verl_project)
`<a href="https://join.slack.com/t/verl-project/shared_invite/zt-3c6mc2khw-v0lo6NfDPuFP6OnkrZwfqw"><img src="https://img.shields.io/badge/Slack-verl-blueviolet?logo=slack&amp">``</a>`
`<a href="https://arxiv.org/pdf/2409.19256"><img src="https://img.shields.io/static/v1?label=EuroSys&message=Paper&color=red">``</a>`
[![Documentation](https://img.shields.io/badge/documentation-blue)](https://verl.readthedocs.io/en/latest/)
`<a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><img src="https://img.shields.io/badge/微信-green?logo=wechat&amp">``</a>`

</div>

verl 是一个灵活、高效且生产就绪的大语言模型（LLM）强化学习训练库。

verl 是 **[HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)** 论文的开源版本。

verl 具有以下灵活易用的特点：

- **轻松扩展多样化强化学习算法**：混合控制器编程模型能够灵活表示和高效执行复杂的后训练数据流。用几行代码即可构建 GRPO、PPO 等强化学习数据流。
- **与现有 LLM 基础设施无缝集成**：通过模块化 API 解耦计算和数据依赖，实现与现有 LLM 框架（如 FSDP、Megatron-LM、vLLM、SGLang 等）的无缝集成。
- **灵活的设备映射**：支持将模型灵活部署到不同的 GPU 集合上，实现高效的资源利用和跨不同集群规模的扩展性。
- 与流行的 HuggingFace 模型即开即用集成

verl 具有以下高效特点：

- **最先进的吞吐量**：集成最先进的 LLM 训练和推理引擎，以及最先进的强化学习吞吐量。
- **3D-HybridEngine 实现高效的 Actor 模型重分片**：消除内存冗余，显著减少训练和生成阶段转换期间的通信开销。

</p>

## 最新动态

- [2025/08] verl 在 [PyTorch 专家交流网络研讨会](https://www.youtube.com/watch?v=Vd79NmmqY3Q&t=2s) 上展示。[幻灯片](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl_talk_pytorch_2025_08.pdf) 已发布。
- [2025/07] [ReTool](https://arxiv.org/pdf/2504.11536) 配方完全开源。[博客](https://www.notion.so/verl-reTool-recipe-Using-multi-round-conversations-and-code-sandboxing-to-improve-the-math-of-large-23a8b5b7feba80b386b2e5b5e3c1cde0)
- [2025/07] 首届 verl 聚会将于 7 月 16 日在温哥华 ICML 举行！如果您在 ICML，请[加入我们](https://lu.ma/0ek2nyao)！（仅限现场）
- [2025/06] 使用 Megatron 后端的 verl 支持大型 MoE 模型，如 [DeepSeek-671B 和 Qwen3-235B](https://verl.readthedocs.io/en/latest/perf/dpsk.html)。
- [2025/03] [DAPO](https://dapo-sia.github.io/) 是基于 Qwen2.5-32B 预训练模型的开源 SOTA 强化学习算法，在 AIME 2024 上达到 50 分，超越了之前由 DeepSeek 的 GRPO（DeepSeek-R1-Zero-Qwen-32B）实现的 SOTA。DAPO 的训练完全由 verl 提供支持，复现代码现在可在 `recipe/dapo` 中找到。

<details><summary> 更多... </summary>
<ul>
  <li>[2025/04] [Seed-Thinking-v1.5](https://github.com/ByteDance-Seed/Seed-Thinking-v1.5/blob/main/seed-thinking-v1.5.pdf) 技术报告发布！使用 verl 训练的 Seed-Thinking-v1.5 在 AIME 2024 上达到 86.7 分，在 Codeforces 上达到 55.0 分，在 GPQA 上达到 77.3 分，在 STEM 和编程方面展现出卓越的推理能力。除了推理任务外，该方法在多样化领域展现出显著的泛化能力。</li>
  <li>[2025/07] verl 在 7/8 的 [AWS AI Hours Singapore](https://pages.awscloud.com/aws-ai-hours-sg.html#agenda) 上做主题演讲，verl & verl-agent 项目更新在 7/11 由 LF AI & Data Singapore 举办的 [Agent for SWE 聚会](https://lu.ma/e498qhsi) 上。</li>
  <li>[2025/06] verl 团队将于 6 月 7 日在 [PyTorch Day China](https://www.lfasiallc.com/pytorch-day-china/) 提供最新项目更新。在北京与我们的开发团队见面！</li>
  <li> [2025/04] [VAPO](https://arxiv.org/pdf/2504.05118)（基于价值的增强 PPO）论文涵盖了我们用于推理模型的最新强化学习方法。从 Qwen-32B-base 模型训练，VAPO 在 AIME 2024 上达到 60.4 分，超越了 DAPO-32B。</li>
  <li>[2025/05] [PF-PPO](https://arxiv.org/abs/2409.06957) 被 ICML 2025 接收，现已在 verl 中支持！PF-PPO 通过过滤潜在噪声奖励信号并通过重放缓冲区重用高质量经验，提升策略学习效率和鲁棒性。</li>
  <li>[2025/04] 我们将在 [ICLR 2025 Expo](https://iclr.cc/virtual/2025/calendar?filter_events=Expo+Talk+Panel&filter_rooms=)、[SCI-FM workshop](https://open-foundation-model.github.io/) 和 [LMSys afterparty](https://lu.ma/d23nyynm) 上举办关于最新后训练技术和 verl 编程指南的教程。演讲材料可在[这里](https://github.com/eric-haibin-lin/verl-community/tree/main/iclr25) 找到。</li>
  <li>[2025/03] verl v0.3.0.post1 发布！查看[发布说明](https://github.com/volcengine/verl/releases/) 了解详情。与之前版本相比实现了[~1.4x 加速](https://tongyx361.github.io/blogs/posts/verl-intro/#/verl-flexible-and-efficient-rl-for-llms)。</li>
  <li>[2025/05] verl 将在 5/16 - 5/17 的 [A2M Shanghai](https://a2m.msup.com.cn/home/?aid=4488&city=shanghai) 上展示。</li>
  <li>[2025/05] verl 将在 [GOSIM x PyTorch Day 2025](https://paris2025.gosim.org/) 上展示。巴黎见！</li>
  <li>[2025/03] 我们在 [vLLM 北京聚会](https://mp.weixin.qq.com/s/n77GibL2corAtQHtVEAzfg) 上介绍了 verl 的编程模型，并在 3 月中旬桑尼维尔的 [SGLang-LMSYS Org 聚会](https://lu.ma/ntjrr7ig) 上进行了 [verl 介绍和更新](https://github.com/eric-haibin-lin/verl-community/blob/main/slides/verl-lmsys-meetup.pdf)。</li>
  <li>[2025/03] 我们将在 EuroSys 2025 上展示 verl（HybridFlow）。鹿特丹见！</li>
  <li>[2025/02] verl v0.2.0.post2 发布！</li>
  <li>[2025/02] 我们在 <a href="https://lu.ma/ji7atxux">Bytedance/NVIDIA/Anyscale Ray 聚会</a> 上展示了 verl。圣何塞见！</li>
  <li>[2025/01] [Doubao-1.5-pro](https://team.doubao.com/zh/special/doubao_1_5_pro) 发布，在 LLM & VLM 上达到 SOTA 级别性能。使用 verl 训练的强化学习扩展预览模型在数学基准测试中达到 OpenAI O1 级别性能（AIME 上 70.0 pass@1）。</li>
  <li>[2024/12] verl 在 Ray Forward 2024 上展示。幻灯片可在<a href="https://github.com/eric-haibin-lin/verl-community/blob/main/slides/Ray_Forward_2024_%E5%B7%AB%E9%94%A1%E6%96%8C.pdf">这里</a> 找到</li>
  <li>[2024/12] 团队在 NeurIPS 2024 上展示了 <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">Post-training LLMs: From Algorithms to Infrastructure</a>。<a href="https://github.com/eric-haibin-lin/verl-data/tree/neurips">幻灯片</a> 和 <a href="https://neurips.cc/Expo/Conferences/2024/workshop/100677">视频</a> 已发布。</li>
  <li>[2024/10] verl 在 Ray Summit 上展示。<a href="https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37">Youtube 视频</a> 已发布。</li>
  <li>[2024/08] HybridFlow（verl）被 EuroSys 2025 接收。</li>
</ul>   
</details>

## 核心特性

- **FSDP**、**FSDP2** 和 **Megatron-LM** 用于训练。
- **vLLM**、**SGLang** 和 **HF Transformers** 用于 rollout 生成。
- 兼容 Hugging Face Transformers 和 Modelscope Hub：[Qwen-3](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_qwen3-8b.sh)、Qwen-2.5、Llama3.1、Gemma2、DeepSeek-LLM 等
- 监督微调。
- 强化学习，支持 [PPO](examples/ppo_trainer/)、[GRPO](examples/grpo_trainer/)、[GSPO](recipe/gspo/)、[ReMax](examples/remax_trainer/)、[REINFORCE++](https://verl.readthedocs.io/en/latest/examples/config.html#algorithm)、[RLOO](examples/rloo_trainer/)、[PRIME](recipe/prime/)、[DAPO](recipe/dapo/)、[DrGRPO](recipe/drgrpo)、[KL_Cov &amp; Clip_Cov](recipe/entropy) 等。
  - 支持基于模型的奖励和基于函数的奖励（可验证奖励），用于数学、[编程](https://github.com/volcengine/verl/tree/main/recipe/dapo) 等
  - 支持视觉语言模型（VLM）和 [多模态强化学习](examples/grpo_trainer/run_qwen2_5_vl-7b.sh)，使用 Qwen2.5-vl、Kimi-VL
  - [多轮工具调用](https://github.com/volcengine/verl/tree/main/examples/sglang_multiturn)
- LLM 对齐配方，如 [自对弈偏好优化（SPPO）](https://github.com/volcengine/verl/tree/main/recipe/sppo)
- Flash attention 2、[序列打包](examples/ppo_trainer/run_qwen2-7b_seq_balance.sh)、[序列并行](examples/ppo_trainer/run_deepseek7b_llm_sp2.sh) 支持，通过 DeepSpeed Ulysses、[LoRA](examples/sft/gsm8k/run_qwen_05_peft.sh)、[Liger-kernel](examples/sft/gsm8k/run_qwen_05_sp2_liger.sh)。
- 通过[专家并行](https://github.com/volcengine/verl/pull/1467) 扩展到 671B 模型和数百个 GPU
- 多 GPU [LoRA 强化学习](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html) 支持以节省内存。
- 使用 wandb、swanlab、mlflow 和 tensorboard 进行实验跟踪。

## 即将推出的功能和变更

- Q3 路线图 https://github.com/volcengine/verl/issues/2388
- 使用 Megatron 的 DeepSeek 671b 优化 https://github.com/volcengine/verl/issues/1033
- 多轮 rollout 和工具使用优化 https://github.com/volcengine/verl/issues/1882
- [智能体集成](https://github.com/volcengine/verl/tree/main/verl/experimental/agent_loop)
- 异步和离策略架构 https://github.com/volcengine/verl/pull/2231
- 自 v0.4 以来的破坏性变更列表 https://github.com/volcengine/verl/discussions/2270

## 快速开始

`<a href="https://verl.readthedocs.io/en/latest/index.html"><b>`文档`</b></a>`

**快速开始：**

- [安装](https://verl.readthedocs.io/en/latest/start/install.html)
- [快速开始](https://verl.readthedocs.io/en/latest/start/quickstart.html)
- [编程指南](https://verl.readthedocs.io/en/latest/hybrid_flow.html) & [技术讲座](https://hcqnc.xetlk.com/sl/3vACOK)（中文）
- [verl 中的 PPO](https://verl.readthedocs.io/en/latest/algo/ppo.html)
- [verl 中的 GRPO](https://verl.readthedocs.io/en/latest/algo/grpo.html)

**逐步运行 PPO 示例：**

- [为后训练准备数据](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
- [为数据集实现奖励函数](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- [PPO 示例架构](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
- [配置说明](https://verl.readthedocs.io/en/latest/examples/config.html)

**可复现的算法基线：**

- [在编程、数学上的强化学习性能](https://verl.readthedocs.io/en/latest/algo/baseline.html)

**代码解释和高级用法（扩展）：**

- PPO 训练器和工作者

  - [PPO Ray 训练器](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP 后端](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM 后端](https://verl.readthedocs.io/en/latest/index.html)
- 高级用法和扩展

  - [使用 FSDP 后端添加模型](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [使用 Megatron-LM 后端添加模型](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)
  - [多轮 Rollout 支持](https://verl.readthedocs.io/en/latest/sglang_multiturn/multiturn.html)
  - [搜索工具集成](https://verl.readthedocs.io/en/latest/sglang_multiturn/search_tool_example.html)
  - [沙盒融合集成](https://verl.readthedocs.io/en/latest/examples/sandbox_fusion_example.html)
  - [使用独立 GPU 资源部署](https://github.com/volcengine/verl/tree/main/examples/split_placement)
  - [扩展到其他强化学习（HF）算法](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [Ray API 设计教程](https://verl.readthedocs.io/en/latest/advance/placement.html)

**社区博客**

- [当推理模型破坏分词时：多轮训练的隐藏复杂性](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/fast_tokenization/multiturn_tokenization_and_masking.md)
- [在 AWS SageMaker 上部署 verl](https://medium.com/@kaige.yang0110/run-verl-on-sagemaker-using-4x8-l40s-gpus-8e6d5c3c61d3)
- [verl x SGLang 多轮代码详解](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme_EN.md)
- [在 verl 中优化 SGLang 内存使用](https://hebiao064.github.io/rl-memory-management)
- [SGLang、verl、OpenBMB 和清华大学：开创端到端多轮 RLHF](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/verl-multiturn-rollout-Release.md)
- [在 AMD GPU 上使用 verl 和 ROCm 集成进行人类反馈强化学习](https://rocm.blogs.amd.com/artificial-intelligence/verl-large-scale/README.html)
- [veMLP x verl：玩转强化学习训练](https://mp.weixin.qq.com/s/7nbqxk4knMGd-hQE9ls2tA)
- [使用 verl 进行 GRPO 分布式强化学习训练最佳实践](https://www.volcengine.com/docs/6459/1463942)
- [HybridFlow verl 原文浅析](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/readme.md)
- [最高提升 20 倍吞吐量！豆包大模型团队发布全新 RLHF 框架，现已开源！](https://team.doubao.com/en/blog/%E6%9C%80%E9%AB%98%E6%8F%90%E5%8D%8720%E5%80%8D%E5%90%9E%E5%90%90%E9%87%8F-%E8%B1%86%E5%8C%85%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9B%A2%E9%98%9F%E5%8F%91%E5%B8%83%E5%85%A8%E6%96%B0-rlhf-%E6%A1%86%E6%9E%B6-%E7%8E%B0%E5%B7%B2%E5%BC%80%E6%BA%90)

## 性能调优指南

性能对于在线策略强化学习算法至关重要。我们编写了详细的[性能调优指南](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) 来帮助您优化性能。

## 升级到 vLLM >= v0.8.2

当使用 FSDP 作为训练后端时，verl 现在支持 vLLM>=0.8.2。请参考[此文档](https://github.com/volcengine/verl/blob/main/docs/README_vllm0.8.md) 了解安装指南和更多信息。请避免使用 vllm 0.7.x，它包含可能导致 OOM 和意外错误的 bug。

## 使用最新 SGLang

SGLang 与 verl 完全兼容，SGLang RL 团队正在广泛构建独特功能，包括多轮智能体强化学习、VLM RLHF、基于服务器的强化学习和部分 rollout。请参考[此文档](https://verl.readthedocs.io/en/latest/workers/sglang_worker.html) 了解安装指南和更多信息。

## 升级到 FSDP2

verl 完全拥抱 FSDP2！FSDP2 由 torch 分布式团队推荐，提供更好的吞吐量和内存使用，并与其他功能（如 torch.compile）可组合。要启用 FSDP2，只需使用 verl main 并设置以下选项：

```
actor_rollout_ref.ref.strategy=fsdp2
actor_rollout_ref.actor.strategy=fsdp2
critic.strategy=fsdp2 
reward_model.strategy=fsdp2 
```

此外，FSDP2 CPU 卸载与梯度累积兼容。您可以通过 `actor_rollout_ref.actor.fsdp_config.offload_policy=True` 开启它以节省内存。更多详情，请参见 https://github.com/volcengine/verl/pull/1026

## AMD 支持（ROCm 内核）

verl 现在支持 FSDP 作为训练引擎（Megatron 支持即将推出），并与 vLLM 和 SGLang 作为推理引擎集成。请参考[此文档](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_build_dockerfile_page.rst) 了解安装指南和更多信息，以及[此文档](https://github.com/volcengine/verl/blob/main/docs/amd_tutorial/amd_vllm_page.rst) 了解 ROCm 的 vLLM 性能调优。

## 引用和致谢

如果您发现该项目有帮助，请引用：

- [HybridFlow: A Flexible and Efficient RLHF Framework](https://arxiv.org/abs/2409.19256v2)
- [A Framework for Training Large Language Models for Code Generation via Proximal Policy Optimization](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2409.19256}
}
```

verl 的灵感来自 Nemo-Aligner、Deepspeed-chat 和 OpenRLHF 的设计。该项目由 Bytedance、Anyscale、LMSys.org、[阿里巴巴 Qwen 团队](https://github.com/QwenLM/)、上海 AI 实验室、清华大学、UC Berkeley、UCLA、UIUC、香港大学、ke.com、[All Hands AI](https://www.all-hands.dev/)、[ModelBest](http://modelbest.cn/)、京东 AI 实验室、微软研究院、[StepFun](https://www.stepfun.com/)、亚马逊、LinkedIn、美团、[Camel-AI](https://www.camel-ai.org/)、[OpenManus](https://github.com/OpenManus)、小米、NVIDIA 研究院、[Baichuan](https://www.baichuan-ai.com/home)、[RedNote](https://www.xiaohongshu.com/)、[SwissAI](https://www.swiss-ai.org/)、[Moonshot AI (Kimi)](https://www.moonshot-ai.com/)、百度、Snowflake、Skywork.ai、JetBrains、[IceSword Lab](https://www.iceswordlab.com) 等众多机构采用和贡献。

## 使用 verl 的优秀工作

- [TinyZero](https://github.com/Jiayi-Pan/TinyZero)：**DeepSeek R1 Zero** 推理任务配方的复现 ![GitHub Repo stars](https://img.shields.io/github/stars/Jiayi-Pan/TinyZero)
- [SkyThought](https://github.com/NovaSky-AI/SkyThought)：NovaSky AI 团队对 Sky-T1-7B 的强化学习训练。![GitHub Repo stars](https://img.shields.io/github/stars/NovaSky-AI/SkyThought)
- [simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)：SimpleRL-Zoo：调查和驯服野外开放基础模型的零强化学习 ![GitHub Repo stars](https://img.shields.io/github/stars/hkust-nlp/simpleRL-reason)
- [Easy-R1](https://github.com/hiyouga/EasyR1)：**多模态** 强化学习训练框架 ![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/EasyR1)
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL)：用于多智能体环境的 LLM 智能体强化学习调优框架。![GitHub Repo stars](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)
- [rllm](https://github.com/agentica-project/rllm)：使用 [verl-pipeline](https://github.com/agentica-project/verl-pipeline) 的异步强化学习训练 ![GitHub Repo stars](https://img.shields.io/github/stars/agentica-project/rllm)
- [RAGEN](https://github.com/ZihanWang314/ragen)：通用推理**智能体**训练框架 ![GitHub Repo stars](https://img.shields.io/github/stars/ZihanWang314/ragen)
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1)：具有推理和**搜索（工具调用）**交错 LLM 的强化学习 ![GitHub Repo stars](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1)
- [ReSearch](https://github.com/Agent-RL/ReSearch)：通过强化学习学习为 LLM **重新**推理和**搜索** ![GitHub Repo stars](https://img.shields.io/github/stars/Agent-RL/ReSearch)
- [Skywork-OR1](https://github.com/SkyworkAI/Skywork-OR1)：Skywork 开放推理器系列 ![GitHub Repo stars](https://img.shields.io/github/stars/SkyworkAI/Skywork-OR1)
- [ToRL](https://github.com/GAIR-NLP/ToRL)：扩展工具集成强化学习 ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/ToRL)
- [Absolute Zero Reasoner](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)：[用于推理的无人类策划数据自对弈框架](https://arxiv.org/abs/2505.03335) ![GitHub Repo stars](https://img.shields.io/github/stars/LeapLabTHU/Absolute-Zero-Reasoner)
- [verl-agent](https://github.com/langfengQ/verl-agent)：**长视野 LLM/VLM 智能体**的可扩展训练框架，以及新算法 **GiGPO** ![GitHub Repo stars](https://img.shields.io/github/stars/langfengQ/verl-agent)
- [RL-Factory](https://github.com/Simple-Efficient/RL-Factory)：智能体学习的简单高效强化学习后训练框架 ![GitHub Repo stars](https://img.shields.io/github/stars/Simple-Efficient/RL-Factory)
- [ReTool](https://retool-rl.github.io/)：ReTool：LLM 中战略工具使用的强化学习。代码发布正在进行中...
- [verl-tool](https://github.com/TIGER-AI-Lab/verl-tool)：基于 verl 的统一且易于扩展的工具智能体训练框架！![GitHub Repo stars](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)
- [PRIME](https://github.com/PRIME-RL/PRIME)：通过隐式奖励进行过程强化 ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/PRIME)
- [MemAgent](https://github.com/BytedTsinghua-SIA/MemAgent)：MemAgent：使用基于多轮强化学习的记忆智能体重塑长上下文 LLM ![GitHub Repo stars](https://img.shields.io/github/stars/BytedTsinghua-SIA/MemAgent)
- [POLARIS](https://github.com/ChenxinAn-fdu/POLARIS)：高级推理模型上扩展强化学习的后训练配方 ![GitHub Repo stars](https://img.shields.io/github/stars/ChenxinAn-fdu/POLARIS)
- [GUI-R1](https://github.com/ritzz-ai/GUI-R1)：**GUI-R1**：用于 **GUI 智能体** 的通用 R1 风格视觉语言动作模型 ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/GUI-R1)
- [DeepRetrieval](https://github.com/pat-jj/DeepRetrieval)：使用**搜索/检索结果**进行**搜索智能体**的强化学习训练 ![GitHub Repo stars](https://img.shields.io/github/stars/pat-jj/DeepRetrieval)
- [Code-R1](https://github.com/ganler/code-r1)：使用可靠奖励为**代码**复现 R1 ![GitHub Repo stars](https://img.shields.io/github/stars/ganler/code-r1)
- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)：通过真实世界环境中的强化学习扩展深度研究 ![GitHub Repo stars](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)
- [VAGEN](https://github.com/RAGEN-AI/VAGEN)：使用多轮强化学习训练 VLM 智能体 ![GitHub Repo stars](https://img.shields.io/github/stars/RAGEN-AI/VAGEN)
- [RM-R1](https://arxiv.org/abs/2505.02387)：推理奖励模型的强化学习训练 ![GitHub Repo stars](https://img.shields.io/github/stars/RM-R1-UIUC/RM-R1)
- [LUFFY](https://arxiv.org/pdf/2504.14945)：在离策略指导下学习推理！![GitHub Repo stars](https://img.shields.io/github/stars/ElliottYan/LUFFY)
- [DeepMath](https://github.com/zwhe99/DeepMath)：用于数学推理的 DeepMath-103K 数据和系列模型！![GitHub Repo stars](https://img.shields.io/github/stars/zwhe99/DeepMath)
- [PACS](https://github.com/ritzz-ai/PACS)：通过监督学习框架进行 RLVR 的隐式 Actor Critic 耦合 ![GitHub Repo stars](https://img.shields.io/github/stars/ritzz-ai/PACS)
- [强化学习的熵机制](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL)：大语言模型推理强化学习的熵机制！![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL)
- [LLaSA-TTS-GRPO](https://github.com/channel-io/ch-tts-llasa-rl-grpo)：基于 LLASA 模型使用 GRPO 优化进行 TTS 微调 ![GitHub Repo stars](https://img.shields.io/github/stars/channel-io/ch-tts-llasa-rl-grpo)
- [PF-PPO](https://arxiv.org/abs/2409.06957)：基于奖励信号可靠性的 PPO 策略过滤，实现更高效和鲁棒的 RLHF。
- [RACRO](https://github.com/gyhdog99/RACRO2)：通过将其解耦为查询条件字幕和纯文本推理构建多模态推理模型 ![GitHub Repo stars](https://img.shields.io/github/stars/gyhdog99/RACRO2)
- [Agent Lightning](https://github.com/microsoft/agent-lightning)：一个灵活且可扩展的框架，可为任何现有智能体框架实现无缝智能体优化。![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/agent-lightning)
- [VTool-R1](https://github.com/VTOOL-R1/vtool-r1)：VLM 通过多模态工具使用的强化学习学会用图像思考。![GitHub Repo stars](https://img.shields.io/github/stars/VTOOL-R1/vtool-r1)
- [Kimina-Prover-RL](https://github.com/project-numina/kimina-prover-rl/tree/main/recipe/kimina_prover_rl)：基于受 DeepSeek-R1 启发的范式进行形式定理证明的训练管道。
- [RL-PLUS](https://github.com/YihongDong/RL-PLUS)：通过混合策略优化对抗 LLM 在强化学习中的能力边界崩溃。
- [rStar2-Agent](https://github.com/microsoft/rStar)：使用多步工具调用的强化学习进行数学任务，rStar2-Agent-14B 在仅 510 个强化学习训练步骤中达到前沿级数学推理 ![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/rStar)
- [Vision-SR1](https://github.com/zli12321/Vision-SR1)：通过推理分解的自奖励视觉语言模型 ![GitHub Repo stars](https://img.shields.io/github/stars/zli12321/Vision-SR1)
- [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL)：SimpleVLA-RL：用于强化学习的简单而有效的视觉语言动作模型 ![GitHub Repo stars](https://img.shields.io/github/stars/PRIME-RL/SimpleVLA-RL)
- [Table-R1](https://github.com/Table-R1/Table-R1)：Table-R1：表格推理的推理时扩展 ![GitHub Repo stars](https://img.shields.io/github/stars/Table-R1/Table-R1)

以及 [recipe](recipe/README.md) 中列出的更多优秀工作。

## 贡献指南

参见[贡献指南](CONTRIBUTING.md)

## 关于 [字节跳动 Seed 团队](https://team.doubao.com/)

成立于 2023 年，字节跳动 Seed 团队致力于打造业界最先进的 AI 基础模型。团队立志成为世界级研究团队，为科学和社会进步做出重大贡献。您可以通过以下渠道更好地了解 Bytedance Seed👇

<div>
  <a href="https://team.doubao.com/">
    <img src="https://img.shields.io/badge/Website-%231e37ff?style=for-the-badge&logo=bytedance&logoColor=white"></a>
  <a href="https://github.com/user-attachments/assets/469535a8-42f2-4797-acdf-4f7a1d4a0c3e">
    <img src="https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white"></a>
 <a href="https://www.xiaohongshu.com/user/profile/668e7e15000000000303157d?xsec_token=ABl2-aqekpytY6A8TuxjrwnZskU-6BsMRE_ufQQaSAvjc%3D&xsec_source=pc_search">
    <img src="https://img.shields.io/badge/Xiaohongshu-%23FF2442?style=for-the-badge&logo=xiaohongshu&logoColor=white"></a>
  <a href="https://www.zhihu.com/org/dou-bao-da-mo-xing-tuan-dui/">
    <img src="https://img.shields.io/badge/zhihu-%230084FF?style=for-the-badge&logo=zhihu&logoColor=white"></a>

</div>
---


---

我们正在招聘！如果您对智能体强化学习的实习/全职机会感兴趣，请发送[邮件](mailto:the.verl.project@gmail.com) 给我们。
