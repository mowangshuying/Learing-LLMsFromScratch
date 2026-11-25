# ch01 - 零基础热身：环境 + PyTorch

> 对应 [README](../README.md) 路线图的 ch01 站。除特别说明外，资料路径均相对于 `3rdparty/LLMs-from-scratch/`。
> 建议用时：1–2 天。文末有「过关标准」自查清单，达标再进 ch02。

## 0. 开始之前

- 本书的 ch01 本身**没有代码**（`ch01/` 下只有 `README.md` 和 `reading-recommendations.md`，讲 LLM 概念与阅读建议），概念部分读一读即可。
- 本站的动手内容实际落在两处：
  - `setup/` —— 环境安装与验证
  - `appendix-A/01_main-chapter-code/` —— PyTorch 速成（part1 / part2 两个 notebook）
- 两个目标：**跑通环境**；**掌握张量、autograd、`nn.Module`、训练循环**。

## 1. 环境搭建（setup/）

### 1.1 选一种安装方式

| 方式 | 资料 | 适合谁 |
|---|---|---|
| uv（作者首推） | `setup/01_optional-python-setup-preferences/README.md` Option 1 | 想体验现代工具链 |
| Conda / Miniforge | 同上 Option 2 | 习惯 conda 生态 |
| pixi（Windows 原生推荐） | `setup/01_optional-python-setup-preferences/native-pixi.md` | Windows 上 `uv add` 遇到依赖问题时 |
| Docker DevContainer | `setup/03_optional-docker-environment/` | 想要完全隔离、一致的环境 |
| 云：Colab / Lightning Studio | `setup/README.md` 的 Cloud Resources | 本机配置弱、想借 GPU |

下面以最通用的 **venv + pip** 为例（Windows PowerShell）。

### 1.2 Python 版本

```powershell
python --version
```

- 推荐 **3.10 及以上**，且比最新版至少旧两个小版本（例如最新是 3.13，就装 3.10 / 3.11），保证 PyTorch 兼容性。
- 没有 Python 的话，Windows 直接从 <https://www.python.org/downloads/> 下载安装。

### 1.3 创建并激活虚拟环境

```powershell
cd D:\WsyWorkSpaces\github\Learing-LLMsFromScratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

- 若 `Activate.ps1` 被执行策略拦截：`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 后重试。
- macOS / Linux 激活命令为 `source .venv/bin/activate`。
- **每开一个新终端都要重新激活**；`.venv/` 已被根目录 `.gitignore` 忽略，不会误提交。
- 用 uv 的等价流程：`pip install uv` → `uv venv --python=python3.10` → 激活 → 后续 `pip` 换成 `uv pip`。

### 1.4 安装依赖

```powershell
pip install -r requirements.txt
```

`requirements.txt` 已放在仓库根目录（与子模块内容一致），各依赖用途：

| 包 | 版本要求 | 用途 |
|---|---|---|
| torch | >= 2.2.2（Intel macOS 限制 < 2.6） | 全部章节的核心框架 |
| jupyterlab | >= 4.0 | notebook 环境 |
| tiktoken | >= 0.5.1 | ch02 / ch04 / ch05 的 BPE 分词器 |
| matplotlib | >= 3.7.1 | ch04 / ch06 / ch07 绘图 |
| tensorflow | >= 2.18.0（Intel macOS >= 2.16.2） | ch05–ch07 仅用于加载 GPT-2 官方权重 |
| tqdm | >= 4.66.1 | ch05 / ch07 进度条 |
| numpy | >= 1.26 | 基础数值依赖 |
| pandas | >= 2.2.1 | ch06 数据处理 |
| psutil | >= 5.9.5 | ch07（torch 会自动带上） |

GPU 说明：Windows 上 pip 安装的 torch 默认自带 CUDA 运行库；需要特定 CUDA 版本时按 <https://pytorch.org> 官网选择器生成安装命令。**ch01–ch04 纯 CPU 笔记本即可**，ch05–ch07 有 GPU 更舒服。

### 1.5 验证环境

```powershell
python 3rdparty/LLMs-from-scratch/setup/02_installing-python-libraries/python_environment_check.py
```

也可以跑一个最小自检：

```python
import sys, torch
print("Python :", sys.version.split()[0])
print("torch  :", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
print("MPS    :", torch.backends.mps.is_available())  # Apple Silicon
```

启动 notebook 环境：

```powershell
jupyter lab
```

- 命令找不到时用全路径：`.venv\Scripts\jupyter-lab`。
- VSCode 用户：直接打开仓库根目录，按 `.vscode` 里的推荐安装 Python / Jupyter 扩展（notebook 工作目录已配置为相对文件路径）。

### 1.6 常见坑

- 综合排查表：子模块根目录 `troubleshooting.md`。
- Jupyter 里包版本对不上 → 多半是 kernel 连错了环境，用 `watermark` 检查。
- JupyterLab 滚动显示异常 → Settings → Settings Editor → 把 "Windowing mode" 改为 `none`。
- Windows 上 uv 装依赖出问题 → 退回普通 `pip`，或改用 pixi。

## 2. PyTorch 速成（appendix-A/01_main-chapter-code/）

| 文件 | 内容 |
|---|---|
| `code-part1.ipynb` | A.1–A.8：张量、计算图与 autograd、`nn.Module`、DataLoader、训练循环、保存/加载 |
| `code-part2.ipynb` | A.9：GPU 使用 |
| `DDP-script.py` / `DDP-script-torchrun.py` | 多 GPU 分布式（选看，本站跳过） |
| `exercise-solutions.ipynb` | 章末练习答案 |

**动手要求：两个 notebook 逐 cell 跟完，再做练习。** 下面是必须吃透的要点。

### 2.1 张量（part1）

```python
import torch

a = torch.tensor([[1., 2.], [3., 4.]])  # shape (2, 2), dtype float32
print(a.shape, a.dtype, a.device)
b = a.reshape(4)      # 展平
c = a * 2 + 1         # 逐元素运算
d = a @ a.T           # 矩阵乘法
e = a.sum(dim=0)      # 按维度归约 -> tensor([4., 6.])
```

关注三件事：**形状（shape）、数据类型（dtype）、所在设备（device）**，后面每一章的 bug 大半出在这三者上。

### 2.2 autograd：`tensor.grad` 是怎么来的（part1）

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x    # y = x² + 3x，dy/dx = 2x + 3
y.backward()
print(x.grad)         # tensor(7.)
```

一句话解释：前向传播时 PyTorch 记录**计算图**；`backward()` 从输出出发按**链式法则**反向求导，把梯度**累加**到叶子张量的 `.grad` 上。由此推出两条纪律：

- 每步更新前要 `zero_grad()`（或 `grad.zero_()`），否则梯度越积越多；
- 推理 / 手动更新参数时用 `with torch.no_grad():`，不建图、省内存。

### 2.3 Dataset / DataLoader 与 nn.Module（part1）

```python
from torch.utils.data import Dataset, DataLoader

class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, i):
        return self.x[i], self.y[i]

loader = DataLoader(ToyDataset(x, y), batch_size=16, shuffle=True)
```

- Dataset 负责「取一条样本」，DataLoader 负责「攒 batch、打乱、并行」。
- `nn.Module` 用 `__init__` 声明层、`forward` 定义前向；参数自动注册，交给 optimizer 管理。

### 2.4 训练循环五步（part1，本站核心）

```text
前向 forward → 算 loss → zero_grad → backward → step
```

### 2.5 保存 / 加载（part1）

```python
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth", weights_only=True))
```

只存 `state_dict`（参数字典），不存整个模型对象——ch05 训自己的小模型时就用这个套路。

### 2.6 GPU（part2）

```python
device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
xb, yb = xb.to(device), yb.to(device)
```

铁律：**模型和数据必须在同一设备**，否则报 device mismatch。

## 3. 动手作业：最小线性回归训练循环

不查资料，独立写一个「前向 → loss → backward → step」完整循环，拟合 `y = 3x + 2`：

```python
import torch
from torch import nn

torch.manual_seed(42)

# 合成数据：y = 3x + 2 + 噪声
x = torch.rand(100, 1)
y = 3 * x + 2 + 0.1 * torch.randn(100, 1)

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 201):
    pred = model(x)          # 1. 前向
    loss = loss_fn(pred, y)  # 2. 算 loss
    optimizer.zero_grad()    # 3. 清旧梯度
    loss.backward()          # 4. 反向求梯度
    optimizer.step()         # 5. 更新参数
    if epoch % 50 == 0:
        print(f"epoch {epoch:3d} | loss {loss.item():.4f}")

print("w =", round(model.weight.item(), 3), "b =", round(model.bias.item(), 3))
# 期望：w ≈ 3，b ≈ 2
```

进阶（检验是否真懂 `.grad`）：不用 optimizer，手动维护 `w`、`b` 两个 `requires_grad=True` 的张量，`backward()` 后在 `no_grad()` 里执行 `w -= lr * w.grad`，再 `w.grad.zero_()`，结果应一致。

## 4. 过关标准自查

- [ ] `python_environment_check.py` 全部通过，JupyterLab / VSCode 能打开 notebook
- [ ] 跟完 `code-part1.ipynb`（A.1–A.8）与 `code-part2.ipynb`（A.9），做完练习
- [ ] **不查资料**能写出「前向 → loss → backward → step」完整循环
- [ ] 能说清 `tensor.grad` 是怎么来的（计算图 + 链式法则 + 累加，为什么要 zero_grad）
- [ ] 独立完成第 3 节线性回归作业，w ≈ 3、b ≈ 2

## 5. 下一站

[ch02 - 文本变张量：分词与数据加载](ch02-文本变张量-分词与数据加载.md)（`ch02/01_main-chapter-code/`，选做 `ch02/05_bpe-from-scratch/`）。
