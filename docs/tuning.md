# Tuning

## 一般模型的训练

- Pre-training（预训练）
  目标：让模型学会语言的基本统计规律、语法、世界知识
  数据：海量无监督文本（万亿 ~ 几十万亿 token）
  目标函数：Next Token Prediction（自回归语言建模）
  这是最耗资源的部分（几千 ~ 几万 H100/A100 天），产出的是 Base 模型（像 LLaMA-3-8B base、Qwen2-7B base）
  → 这个阶段严格来说不算“tuning”，而是 from scratch 训练，但它是所有后续 tuning 的基础。

- Supervised Fine-Tuning（SFT） / Instruction Fine-Tuning（指令微调）
  目标：让模型学会“听懂人类指令 + 按格式输出高质量回答”
  数据：高质量的 instruction-response 对（Alpaca、ShareGPT、UltraChat、OpenHermes、合成数据等）
  目标函数：还是 Next Token Prediction，但只在 curated 的对话/任务数据上继续训练
  产出：Instruct / Chat 模型 的初步版本（像 LLaMA-3-8B-Instruct 第一版）
  → 这是大多数人说的“fine-tuning”的起点，消耗资源少很多（几张卡几天就能出结果）

- Alignment / Preference Tuning（对齐 / 偏好优化）
  目标：让模型输出更符合人类偏好（helpful、honest、harmless），减少 hallucination、毒性、偏见
  主流子方法（2025–2026 年已高度多样化）：

  RLHF（经典）：Reward Model + PPO（ Proximal Policy Optimization ）
  DPO（Direct Preference Optimization）：直接用偏好对优化，最流行（简单、无需 reward model、无需在线采样）
  ORPO（Odds Ratio Preference Optimization）：把 SFT + 偏好对齐合并成一步
  KTO / SimPO / GRPO 等变体

数据：人类（或 AI）标注的 preference pairs（chosen vs rejected 回复）
产出：最终的 对齐版模型（像 ChatGPT、Claude、Grok、LLaMA-3-Instruct 最终版）

## special version LLM like reasoning model

Pre-training（基础预训练，同上）

- Continued Pre-training / Mid-training / Domain-specific Pre-training（继续预训练 / 中间训练）
  在特定领域数据（代码、数学、长上下文、中文等）上继续 next-token 训练
  常用来做“知识注入”或“上下文长度扩展”

Supervised Fine-Tuning (SFT / IFT)（指令微调，同上）

- Reasoning / Thinking Tuning（可选，针对 reasoning 模型）
  用 chain-of-thought、process supervision、verifiable reward 数据强化思考过程
  常见于 o1-style 模型（RLVR、GRPO 等）

Preference / Alignment Tuning（DPO / ORPO / RLHF / RLVR 等）

- Safety / Red-teaming Tuning（可选最后一步）
  专门针对 jailbreak、敏感话题、安全性再做一轮 SFT 或 preference tuning
