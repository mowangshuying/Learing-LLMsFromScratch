# LLMs-from-scratch 学习路线（7 站）

主线：零基础 → 训出自己的小模型 → 分叉进阶。
所有路径均相对于 `3rdparty/LLMs-from-scratch/`，每站附「过关标准」，达标再进下一站。

```
站1 热身 → 站2 数据 → 站3 注意力 → 站4 搭GPT → 站5 预训练★ → 站6 微调 → 站7 分叉进阶
                                                （★ 里程碑：自己的小模型）
```

---

## 站 1｜零基础热身：环境 + PyTorch

- 目标：跑通环境，掌握张量、autograd、`nn.Module`、训练循环。
- 路径：
  - `setup/`（环境安装建议，含 `requirements.txt`）
  - `appendix-A/01_main-chapter-code/`（PyTorch 速成，part1/part2）
  - 遇到问题查 `troubleshooting.md`
- 动手：跟完 appendix-A 两个 notebook，独立写一个最小训练循环（线性回归即可）。
- 过关标准：不查资料能写出「前向 → loss → backward → step」完整循环；说清 `tensor.grad` 是怎么来的。

## 站 2｜文本变张量：分词与数据加载（ch02）

- 目标：理解 embedding、位置编码、滑窗采样和 DataLoader。
- 路径：
  - `ch02/01_main-chapter-code/`（ch02.ipynb + dataloader.ipynb）
  - 选做：`ch02/05_bpe-from-scratch/`（手写 BPE，强烈推荐）
- 动手：对一段自选文本实现「分词 → token id → 输入/目标对 → batch」。
- 过关标准：能解释 input/target 为什么错一位；能手算一个 batch 的形状。

## 站 3｜注意力机制（ch03）

- 目标：从简化 self-attention 一路推到 causal multi-head attention。
- 路径：
  - `ch03/01_main-chapter-code/`（ch03.ipynb + multihead-attention.ipynb）
  - 对照：`ch03/02_bonus_efficient-multihead-attention/`（工程实现版）
- 动手：不看答案手写 `MultiHeadAttention` 类，和书里的输出对数值。
- 过关标准：能解释 Q/K/V、缩放因子 √d、causal mask 各自解决什么问题。

## 站 4｜从零搭 GPT（ch04）

- 目标：把 LayerNorm、GELU、FFN、残差、注意力组装成 124M 参数的 GPT-2 结构。
- 路径：
  - `ch04/01_main-chapter-code/`（ch04.ipynb + gpt.py）
- 动手：跑通 dummy 前向（输出形状 `[batch, seq, 50257]`），再加载 GPT-2 官方权重生成文本。
- 过关标准：能画出数据在模型里的完整流向；数清 124M 参数分布在哪几类层里。

## 站 5｜预训练：训出自己的小模型 ★（ch05 + appendix-D）

- 目标：完成本书核心里程碑——在语料上预训练自己的 GPT，并加载回 GPT-2 权重对比。
- 路径：
  - `ch05/01_main-chapter-code/`（ch05.ipynb + gpt_train.py + gpt_generate.py）
  - `appendix-D/01_main-chapter-code/`（学习率预热、余弦衰减、梯度裁剪）
  - 选做：`ch05/03_bonus_pretraining_on_gutenberg/`（换更大语料）
- 动手：完整跑一轮预训练，观察 loss 曲线；对比 temperature / top-k 采样效果。
- 过关标准：拿到一个自己训的 checkpoint，能生成连贯（哪怕幼稚）的文本；能解释为什么 loss 降但生成仍差。

## 站 6｜让模型有用：两种微调（ch06 + ch07）

- 目标：掌握下游微调的两条路——分类头微调、指令微调。
- 路径：
  - `ch06/01_main-chapter-code/`（垃圾邮件分类微调）
  - `ch07/01_main-chapter-code/`（指令微调 + `ollama_evaluate.py` 自动评估）
  - 选做：`appendix-E/01_main-chapter-code/`（LoRA 参数高效微调）、`ch07/04_preference-tuning-with-dpo/`（DPO 偏好对齐）
- 动手：各跑通一次微调；用 Ollama 给指令模型的回答打分。
- 过关标准：能说清「预训练 / 分类微调 / 指令微调」三者在数据格式和损失上的区别；LoRA 为什么省显存。

## 站 7｜分叉进阶（按兴趣选一条或多条）

- 分支 A：现代架构演进（推荐接续站 5）
  - `ch05/07_gpt_to_llama/`（GPT → Llama：RMSNorm、SiLU、RoPE）
  - `ch05/11_qwen3/`、`ch05/12_gemma3/`、`ch05/13_olmo3/`（从零实现当代开源模型）
- 分支 B：推理与效率（偏工程）
  - `ch04/03_kv-cache/`、`ch04/04_gqa/`、`ch04/05_mla/`、`ch04/06_swa/`、`ch04/07_moe/`
- 分支 C：推理模型（reasoning）
  - `reasoning-from-scratch/` 是嵌套子模块（指向 rasbt/reasoning-from-scratch），当前为空。
  - 启用：在仓库根目录执行 `git submodule update --init --recursive`
- 过关标准（任选其一）：从零实现一个 Qwen3/Llama 结构并加载官方权重；或给站 5 的模型加上 KV cache 并测出生成提速；或跟完 reasoning 仓库的一个完整训练示例。

---

## 使用建议

- 每章先跑 `01_main-chapter-code` 主线 notebook，再做章末练习（答案在各章 `exercise-solutions.ipynb`）。
- bonus 目录不必全做，路线里已挑出性价比最高的。
- 站 1–4 普通笔记本即可；站 5–6 有 GPU 更舒服，小模型 CPU/MPS 也能跑（见 `ch05/19_cpu_mps_differences/`）。