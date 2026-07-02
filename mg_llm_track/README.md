# MG LLM Track — 最简版本："LLM + 时间模块"核心机制验证

**目的**：不动 CARE / AnomLLM 两条重线，用最省事的合成数据（Mackey-Glass）单独跑通并验证核心架构——

```
MG 时间序列 → LAFR（时间模块，自监督预训练，随后冻结）
            → adapter 把 LAFR 输出对齐成 token
            → 拼接问题文本 → 冻结 Qwen3.5-0.8B → 答案
```

即：**LLM 是被动的冻结解读器，序列永远先过时间模块，"对齐"发生在 adapter 训练阶段，不是 LLM 推理时主动去调用模块**（完整论证见 [../PAPER_PLAN.md](../PAPER_PLAN.md) §0.5）。这是 2026-07-01 拍板的核心方案：统一入口 + 内部路由，骨干统一为 Qwen3.5-0.8B。

---

## 这个文件夹里有什么

只有本 README + 运行结果的存档（结果生成后回填）。**可执行脚本没有放在这里**，原因见下方"依赖说明"。

## 依赖说明（为什么脚本不在这个文件夹）

真正跑实验的脚本是 [`../experiments/mg_llm_adapter.py`](../experiments/mg_llm_adapter.py)。它复用了三个已有的共享模块，这些模块也被 CARE 线的脚本用着，不应该复制一份出来（会导致后续修改两边不同步）：

| 复用的模块 | 提供什么 |
|---|---|
| [`experiments/lafr_encoder.py`](../experiments/lafr_encoder.py) | LAFR 编码器本体 |
| [`experiments/lafr_eval.py`](../experiments/lafr_eval.py) | `train_lafr`（自监督预训练）、`fit_standardizer`/`standardize` |
| [`experiments/llm_adapter.py`](../experiments/llm_adapter.py) | `TemporalAdapter`、`FrozenLLM`、`make_qa`（三重接地对照协议：correct/shuffled/no_token）、`train_adapter`、`answer_acc` |
| [`experiments/mg_lafr_forecast.py`](../experiments/mg_lafr_forecast.py) | `mackey_glass()` 合成序列生成器 |

`mg_llm_adapter.py` 只新增了两件事：① 用独立随机种子/延迟生成多通道 MG "数据库"并从中切窗口（替代 CARE 真实 SCADA 数据）；② 把上面几个模块串成一条完整管线，命令行入口换成 `--llm` 默认 `Qwen/Qwen3.5-0.8B`。

## 怎么运行

**必须在项目根目录（`F:\论文\TNN-LLM\LLTM`）下执行**，因为脚本按相对路径找模型和输出目录：

```bash
# 1. 冒烟测试（约1-2分钟，验证管线跑通，不代表真实信号）
python experiments/mg_llm_adapter.py --smoke

# 2. 正式运行（笔记本单卡 GPU，显存约需 5-6GB，耗时视 GPU 而定）
python experiments/mg_llm_adapter.py
```

常用可调参数（全部有默认值，不传就用默认）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--llm` | `Qwen/Qwen3.5-0.8B` | 冻结骨干模型 ID |
| `--channels` | 8 | MG 合成的通道数 |
| `--window` | 72 | 每个窗口的时间步数 |
| `--patch` | 6 | LAFR patch 大小 |
| `--lafr-windows` | 600 | LAFR 自监督预训练用的窗口数 |
| `--lafr-epochs` | 6 | LAFR 预训练轮数 |
| `--n-train` / `--n-eval` | 300 / 180 | adapter 训练/评估用的 QA 样本数 |
| `--epochs` | 8 | adapter 训练轮数 |
| `--batch-size` | 8 | adapter 训练/评估 batch（显存紧张可调小，宽裕可调大加速） |

## 输出在哪、怎么看

结果写到 `results/mg_llm_adapter/`（`--smoke` 会多套一层 `smoke/` 子目录）：

- `mg_llm_adapter_report.json` — 训练曲线 + pre/post-train 三条件准确率 + 按任务（lead/conf/cp）细分
- `lafr_mg_qa.pt` — 本次训练用的冻结 LAFR 编码器权重
- `adapter.pt` — 训练好的 adapter 权重

**判定标准（接地是否成立）**：`correct` 准确率必须显著高于 `shuffled` 和 `no_token`（脚本里用 gap > 0.1 作为"grounded"的硬阈值）。三者打平说明 LLM 没有真正用上时间模块给的结构信息，只是在瞎猜或靠语言先验答题。

## 与大论文计划的关系

这是 [`../PAPER_PLAN.md`](../PAPER_PLAN.md) 里 CARE 线（C1/C2，真实 SCADA 数据，问答接地）的**简化替身**：同一套接地协议、同一个 adapter 架构，只是把真实 CARE 窗口换成构造已知、ground truth 精确的 MG 合成窗口，且骨干统一到 Qwen3.5-0.8B。跑通且接地成立，可以先作为"机制在最简条件下成立"的验证性结果/附录材料；CARE 真实数据线仍然是**独立于本实验的、需要单独重跑**的任务，不因为这里跑通了就算完成。

---

## 运行结果（回填）

<!-- RESULTS_PENDING -->
本次正式运行（非 smoke）于 2026-07-01 提交后台执行，配置见上表默认值。结果回填时会替换本节。
