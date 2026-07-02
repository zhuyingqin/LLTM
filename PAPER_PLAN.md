# Paper Plan

**候选标题**（英文，投稿用）：
1. *Beyond Serialized Numbers: Grounding Frozen LLMs in Learned Lag/Relation Structure for Multivariate Time Series*
2. *LAFR: Lag-Aware Relational Tokens for Frozen-LLM Time-Series Understanding* ← **2026-07-01 GPT-5.5 评审推荐用这个**
3. *Structure, Not Strings: Teaching Frozen LLMs to Read Multivariate Time-Series Lag and Dependency Structure*

**一句话贡献**：把一个自监督学到的、显式建模变量间有向滞后与条件依赖结构的时序编码器（LAFR）以 token 形式接入冻结 LLM——而不是把同一段窗口序列化成文本——无论是做结构化问答（用 shuffled/no-token 对照验证是真接地而非瞎猜）还是做异常区间定位（在 AnomLLM 标准合成基准上验证），都比纯文本 LLM 基线、也比编码器自己单独输出的检测结果更好。

**目标venue**：ICLR（skill 默认值，未指定 `— venue:` 覆盖；见文末"关于venue的建议"）
**论文类型**：Method paper（偏系统/机制诊断）
**日期**：2026-06-30（2026-07-01 更新①：新增 §0.5 架构数据流定调；C4 升级为"部分支持"，纳入 `results/mg_lafr_forecast/` 新落盘结果。2026-07-01 更新②：架构决策定为"统一入口+内部路由"，冻结骨干统一为 Qwen3.5-0.8B，新增 C6 "structure beats scale" 对照）
**页数预算**：9页（正文到 Conclusion 结束，不含参考文献/附录）
**章节数**：7

---

## 0. 证据盘点（写作前必读）

在写 Claims-Evidence Matrix 之前，先说清楚"哪些是真实验证过的数字、哪些只是代码/协议已经写好但没有保存结果"——这是本计划最重要的前提，直接决定了论文现在能不能写、先写哪部分。

| 实验线 | 代码位置 | 本工作区内是否有保存的结果文件 | 结论 |
|---|---|---|---|
| CARE 真实风场 SCADA：机制评估（变化点定位、超前滞后恢复 vs 经典基线） | [experiments/lafr_eval.py](experiments/lafr_eval.py) | **没有**（脚本支持 `--output-dir` 写 `lafr_eval_report.json`，但本仓库里不存在任何 `results/` 目录） | 协议已实现，数字不存在，**需要重跑** |
| CARE 问答接地：text_zeroshot / text_soft / adapter × correct/shuffled/no_token | [experiments/llm_adapter.py](experiments/llm_adapter.py)、[experiments/llm_compare.py](experiments/llm_compare.py) | **没有**（同样支持 `--output-dir` 写 json，但目录不存在） | 协议已实现，数字不存在，**需要重跑** |
| CoT 公平性检查 / 定性自由生成 | [experiments/llm_cot_compare.py](experiments/llm_cot_compare.py)、[experiments/llm_qualitative.py](experiments/llm_qualitative.py)、[experiments/llm_fulltext.py](experiments/llm_fulltext.py) | 无保存结果，且这三个脚本设计上就是打印到终端，不落盘 | 只能作为附录里的定性展示，**不能**作为主结果 |
| lag 读出策略消融 | [experiments/lag_readout_probe.py](experiments/lag_readout_probe.py) | 无保存结果（脚本本身也不写文件，只 print） | **需要**改成落盘再重跑 |
| AnomLLM 合成基准：llm / lafr_llm / lafr_llm_edit / lafr_detector | [anomllm-compare/run_lafr_qwen_compare.py](anomllm-compare/run_lafr_qwen_compare.py) 产出的 5 个真实 json | **有**：`compare_results.json`、`lafr_qwen_compare_results.json`、`lafr_qwen_token_policy_compare_results.json`、`lafr_qwen_edit_compare_results.json`、`lafr_qwen_trend_policy_results.json` | **可直接写进论文的定量结果** |
| LAFR forecast head 独立预测：Mackey-Glass 合成序列（由单变量构造 5 通道视图：x/lag1/lag6/ma7/dx），24 patch 上下文 → 4 patch 预测 | [experiments/mg_lafr_forecast.py](experiments/mg_lafr_forecast.py) | **有**：`results/mg_lafr_forecast/metrics.json`（含 35 epoch 训练曲线 + 测试集 522 窗口指标）、两张图、`lafr_mg_forecast.pt` 权重 | **真实落盘结果**（2026-07-01 新增）：LAFR MSE 0.0174 / R²=0.982，naive 基线 MSE 1.469 / R²=−0.54。但只对比了 naive、只在合成数据上 → 只能作为"forecast head 功能性验证"的辅助证据，不能当独立预测贡献 |

**结论：现在手上真正"可引用"的数字有两条线——AnomLLM 合成基准（主结果）和 Mackey-Glass forecast head 功能性验证（辅助证据）。** CARE 真实数据这条线（问答接地 + 机制评估）代码质量很高、协议设计得也很讲究（三重接地对照、CoT 公平性检查这些都是很扎实的实验设计），但数字全部缺失，必须重跑才能写进 Results。这不是否定这条线，而是提醒：**在启动 `/paper-write` 之前，必须先把 CARE 这条线的实验重新跑一遍并把 json 保存下来**（其余数据/权重原本大概率在 `C:\杂物\LLM-NN\` 这个更大的工作区里，`anomllm-compare/*.json` 里的 `config.data_root`/`model` 字段还留着这个路径，值得先去那边找一下有没有现成的 `results/lafr_eval_full_fixed/`、`results/llm_adapter_full_qwen3/` 之类的产出，能省一次重跑）。

另外发现一个方法学细节，也一并记在这里：两条线目前用的 Qwen3 规模不一样——CARE 那条线的脚本默认 `Qwen/Qwen3-0.6B`，AnomLLM 那条线用的是 `Qwen3-8B`（4bit）。**2026-07-01 决策：统一到 Qwen3.5-0.8B**（重跑时直接换骨干；接口 B 先 smoke test，细则见 §0.5），实施前先确认 Qwen3.5-0.8B 在 HF 上的准确模型 ID 与许可证。若接口 B 在 0.8B 上确实跑不稳，退路是在 Method/Limitations 里明确说明"两种接入方式对模型规模的敏感度不同"，否则审稿人会怀疑是不是在挑对自己有利的模型规模。

---

## 0.5 架构数据流定调（2026-07-01 新增，回应架构确认问题）

写 §3 Method 和画 Fig 1 之前必须统一口径：本工作的数据流是——

```
时间序列 ──→ LAFR（时间模块，冻结/自监督预训练）──→ 结构化产物
                                                      │
        接口A：adapter 把产物投影成 LLM 嵌入空间的 token（对齐在 adapter 训练阶段完成）
        接口B：产物序列化成统计证据文本 + 候选区间
                                                      │
                             拼上问题文本 ──→ 冻结 LLM ──→ 答案 / 编辑指令
                                            （接口B 再经确定性 executor 落地成最终区间）
```

**不是**"输入先给 LLM、LLM 分析后再把序列拿到时间模块对齐"。三个必须在论文里讲清楚的点：

1. **LLM 是被动的冻结解读器，不是调度者。** LLM 从不调用时间模块（不是 tool-calling/agent 架构），接口 A 里它从不接触原始数值序列，接口 B 里它从不直接输出最终分数。方向画反了，读者会以为这是一篇 LLM-agent 论文——那是另一个故事，当前代码和实验都不支持。
2. **"对齐"发生在训练时，不发生在推理时。** adapter 训练阶段学会把 LAFR 表征投影到 LLM 嵌入空间（BLIP-2/LLaVA 同款配方）；推理时只是一次前向投影，没有任何交互式"拿去对齐"的步骤。
3. **这个方向是三重接地对照（correct/shuffled/no_token）成立的前提。** 正因为注入点受控（token 可以被替换/打乱/移除），才能证伪"LLM 靠语言先验瞎猜"；如果 LLM 自己决定何时调用模块，这个对照就做不干净了。

**2026-07-01 架构决策（用户拍板）**：核心方案定为**"统一入口 + 内部路由"**，统一冻结骨干换成 **Qwen3.5-0.8B**。对外/对用户的叙事可以说"输入交给 LLM 系统，系统把序列拿到时间模块对齐后作答"——这与上图机制完全一致（路由固定、对齐由 adapter 在训练期学会），论文写作仍严格按本节数据流描述，不改成 agent/tool-calling 叙事。配套动作：
- ① CARE 线 adapter 在 Qwen3.5-0.8B 上重训（原 0.6B 协议直接迁移，改动只有骨干）；
- ② 接口 B 的 JSON 编辑先在 0.8B 上 smoke test：稳定 → 两条线统一骨干（顺带解决评审点名的规模不一致问题）；不稳定 → 优先试 JSON schema 受限解码，仍不行则接口 B 保留 8B 并在 Limitations 明确解释；
- ③ 新增"structure beats scale"对比（见 Claims C6）：小冻结模型 + 结构 token 对阵大模型 + 文本序列化，若成立是新卖点。

---

## 1. Claims-Evidence Matrix

| Claim | Evidence | 状态 | 章节 |
|---|---|---|---|
| C1. LAFR（自监督、无需事件标签、通道数无关：逐变量对滞后库 + 滞后感知跨变量注意力 + 可微边界检测器）能从真实多变量 SCADA 窗口中恢复变化点和超前滞后结构 | `lafr_eval.py` 的注入已知结构评估协议（真实 CARE 窗口注入已知 t*/tau，对比 max-jump / 互相关经典基线） | 协议已实现，**无保存结果 → 需重跑** | §3, §4 |
| C2. 用训练好的 adapter 把 LAFR 输出成 event/channel/pair 三组 token（按通道字母显式绑定）接入冻结 LLM，比把同一窗口序列化成文本，更能支撑结构化问答；用 correct/shuffled/no_token 三重对照证明这是真接地，并用一次独立的 CoT 公平性检查排除"文本基线只是没给够思考预算"这个反驳 | `llm_adapter.py` / `llm_compare.py` / `llm_cot_compare.py` | 协议已实现，**无保存结果 → 需重跑** | §3, §4 |
| C3. 让 LLM 以结构化 JSON 编辑操作（DROP/SHIFT/EXPAND/SHRINK/MERGE）修正 LAFR 给出的候选异常区间（由确定性 executor 执行，LLM 不直接输出最终分数），在 AnomLLM 标准合成基准（point/range/trend/freq，官方生成器 + affiliation-F1 指标）上，同时超过纯文本 LLM 基线和不接 LLM 的 LAFR 检测器本身 | **真实数字**（见下方"核心结果"），来自 `anomllm-compare/*.json` | **已验证**，但 n=8/类样本量小，且 trend 策略在训练集上做过校准，需要透明说明 | §5 |
| C4. LAFR 的 `forecast_head`（下一 patch 残差预测）是一个独立可用的"预测"能力 | `results/mg_lafr_forecast/metrics.json`（2026-07-01 新增）：Mackey-Glass 合成序列上 24→4 patch 预测，LAFR R²=0.982 vs naive R²=−0.54 | **部分支持** —— forecast head 功能性已验证（有真实落盘数字），但只对比 naive 基线、只在合成数据上，没有对比标准预测基线（PatchTST / iTransformer），**仍不能作为独立卖点**；可作为 §3 的功能性 sanity check 或附录实验 | §3（一句话+附录）, §6（升级为预测贡献所需的补实验写进 future work） |
| C5. 同一套 token + LLM 解读机制可以进一步给出"控制系统的值"（例如建议的桨距角/转矩调整量） | 仓库中**零代码、零控制标签数据、零仿真环境** | **完全不支持** —— 仅作展望，不放进 Contributions，也不放进 Experiments | §6（仅 Discussion/Outlook） |
| C6.（2026-07-01 新增）"结构胜过规模"：冻结 Qwen3.5-0.8B + LAFR token 在结构化任务上不低于甚至超过更大冻结模型 + 文本序列化 | 计划中的新对比：两条线重跑时在同一任务上产出 {0.8B+token, 0.8B+文本, 8B+文本} 三方对照 | **计划中，无结果** —— 成立则升级为独立卖点，不成立则如实降级为规模敏感性分析放 §6 | §4/§5（数据）, §6（解读） |

**核心结果（来自真实 json，务必在正式写作前扩大样本量、加多 seed）**：

- 四类异常平均 affiliation-F1（`lafr_qwen_token_policy_compare_results.json`，n=8 条/类）：`llm=0.360` → `lafr_llm=0.583` → `lafr_llm_edit=0.814`；对照 `lafr_detector=0.635`（同一 8 条子集上重新计算）。
- trend 单类（最难的一类，`lafr_qwen_trend_policy_results.json`）：`llm=0.125`，`lafr_detector=0.189`，`lafr_llm_edit=0.874`。
- 经典基线 + 纯 LAFR 检测器的对照（`compare_results.json`，**全量** n=120 条/类，与上面两组的 n=8 不是同一口径，写正式结果表前需要统一）：iso_forest 和阈值法在 affiliation-F1 上普遍弱于 LAFR 检测器本身（例如 range 类：iso_forest 0.576 / threshold 0.644 / lafr_ours 0.745）。

---

## 2. 论文结构（7 节，9 页）

### §0 Abstract（150–250 词，正式稿英文；**2026-07-01 评审意见：开头第一二句就给已验证的 AnomLLM 硬数字（0.360→0.814），并加一句 scope note 说明 SCADA 接地实验为协议展示/待补数字——除非届时 CARE 线已重跑完成**）
- **做到了什么**：LAFR——自监督、通道数无关的多变量时序编码器，显式建模逐变量对有向滞后与条件依赖结构；两种接入冻结 LLM 的机制（token 接地问答 / 证据到编辑的异常区间修正）。
- **为什么重要/难**：现有"LLM 用于时间序列"的主流做法把数值直接序列化成文本，既丢失精确的滞后/依赖结构，也没有机制证明 LLM 是真的读懂了数字而不是靠语言先验瞎猜；专用编码器学得到结构，却没有"说出来"或变成可执行判断的出口。
- **怎么做的**：VTE→PLB→LVAA→DBD 的自监督编码器 + 两种冻结 LLM 接口。
- **证据**：AnomLLM 标准合成基准上的真实提升（见核心结果）；CARE 真实数据上的三重接地对照（**待重跑补齐**）。
- **最亮结果**：trend 类异常上，检测器单独 0.189 → LLM 编辑后 0.874（同时标注 n=8，避免摘要里读起来像是无条件的大提升）。
- **自洽性检查**：不读全文也能看懂——是。

### §1 Introduction（1.5页）
- **开场钩子**：把多变量时间序列丢给通用 LLM（预测/问答/异常检测）正变流行，但几乎都是"数值转文本"，LLM 并不天然理解"哪个传感器领先哪个多少步"这类结构。
- **缺口**：文本序列化丢结构，也没有对照实验证明 LLM 用到了数字本身；专用编码器有结构却没有可读出口。
- **一句话贡献**：见上。
- **做法概览**：LAFR + 两种接口——**在这里要明确说这两条实验线是"同一个机制在两种任务/数据上的两次检验"（问答接地 + 异常区间修正），是为了证明机制的通用性，而不是两篇论文硬拼在一起**（这是自查后加的一句关键定位句，否则容易被读成两个故事）。
- **贡献列表（4条，逐条对应 Claims-Evidence Matrix，不超范围）**。**投稿版收窄规则（2026-07-01 评审意见）**：若 CARE 线（C1/C2）在投稿前未完成重跑，则贡献列表只保留已验证主张——第 1、2 条改写为"提出编码器与两种接口的设计+接地协议"（协议贡献，不引用数字），第 4 条整体移出贡献列表、降为 §4 的"protocol presented, results pending"说明；一句话贡献同步收窄到只主张 AnomLLM 已验证结果：
  1. 提出 LAFR：自监督、无标签、通道数无关的滞后/关系感知多变量编码器。
  2. 设计两种接入冻结 LLM 的机制：token 接地适配器（带通道绑定 + 三重接地对照）与证据到编辑的结构化提示范式。
  3. 在 AnomLLM 标准合成基准上证明：LLM 编辑后的方案同时超过纯文本 LLM 基线（+0.22～0.45 affiliation-F1）和不接 LLM 的检测器本身（trend 类 +0.68），且全程复用官方生成器与指标代码保证可比性。
  4. **（标注"结果待重跑"）** 在真实风场 SCADA 数据上，用注入已知结构的机制评估验证 LAFR 学到的确实是可解释的物理结构而非黑箱特征。
- **结果预览**：trend 类 0.189→0.874（标注 n=8）。
- **Hero figure**：见 Figure Plan Fig 1。
- **前置检查**：摘要+引言结尾前已给出核心结论和最强数字 —— 是。
- **关键引用**：文本序列化 LLM4TS 代表作 2-3 篇 [VERIFY]；AnomLLM 基准本身 [VERIFY]；一篇滞后/关系感知多变量架构 [VERIFY]。

### §2 Related Work（1.0页，禁止写成罗列）
四个子类，按方法论家族组织，每类都要说清楚"和我们的假设/输出形式有什么本质不同"。**并以一张 4 行小对比表收尾（2026-07-01 评审意见）**：列 = {先前家族, LLM 看到的输入形式, 滞后/依赖结构是否显式, LAFR 相对多了什么}，行 = 下面 (a)–(d) 四个家族——用表格替代冗长的逐段对比，同时省页数：
- (a) **文本序列化 LLM4TS**（LLMTime、PromptCast、Time-LLM、GPT4TS/One-Fits-All 一类）——它们把 LLM 当通用序列骨干或直接喂数字文本，没有专门的滞后/关系结构建模，也不存在"接地是否真实"的对照。
- (b) **滞后/关系感知的多变量时序表示学习**（Crossformer、iTransformer、TimesNet 一类）——它们学结构但止步于分数/预测输出，没有语言接口。
- (c) **时序异常检测**（经典 iso_forest/threshold，深度方法，以及 AnomLLM 这类"LLM 做检测"的基准本身）——AnomLLM 直接把序列转文本喂 LLM 判断异常区间；我们复用它的生成器和指标，证明"结构化证据 + LLM 编辑"比它默认的纯文本路线更好。
- (d) **冻结骨干 + 可训练适配器**（BLIP-2、LLaVA、prefix-tuning 一类多模态/软提示接地方法）——同样的"冻结骨干+轻量适配器"配方，但我们用在时间序列 token 而非视觉 token 上，并且多做了一层它们通常没有的接地对照实验（shuffled/no_token）。

### §3 Method（2.5页 —— 原 2.0 页；2026-07-01 采纳评审意见把 §6 压到 0.5 页，省出的 0.5 页给方法节，缓解"5 个子模块 + 2 种接口装不下"的风险）
- **记号**：窗口 X∈R^{T×C}，patch 数 P，通道数 C，滞后库 `lag_bins`，事件数 K，关系维度 d_r。
- **LAFR 内部**（建议压成 1 张图 + 每模块 1 段话，详细推导放附录，否则 2 页装不下）：VTE（变量当 token）→ PLB（逐变量对三重积滞后特征）→ LVAA（滞后感知跨变量注意力，带可学习有向滞后）→ DBD（可微软 top-K 边界检测，不用 argmax）→ 输出具名 `LAFROutput`（event / pair-relation / dependency-graph / episode 五类 token 与矩阵）。自监督目标：预测残差 + 掩码重建 + 变化点对比，无需事件标签。
- **接口 A（token 接地）**：event/channel/pair 三组 token，channel token 显式绑定问题里的字母（"A=`<c_0>` B=`<c_1>`"），只训练 adapter，LAFR 和 LLM 全程冻结；接地判据（可证伪）：`correct ≫ shuffled ≈ no_token` 才算通过。
- **接口 B（证据到编辑）**：LAFR 输出候选区间 + 统计证据（峰值、斜率等）→ prompt → LLM 输出结构化编辑操作（DROP/SHIFT/EXPAND/SHRINK/MERGE）→ 确定性 executor 应用到 LAFR 原始分数上——**LLM 从不直接吐出最终数值**，这是解释"为什么比纯 LLM 更稳"的关键设计点，要在方法节讲清楚。
- **页数风险**：编码器 5 个子模块 + 2 种接口，2 页可能装不下所有公式。**建议**：主文给 1 张架构图 + 精简符号表 + 每部分 2-3 句话，把 PLB/LVAA 的具体张量运算公式移到附录。

### §4 Experiments I：真实 SCADA（CARE）上的结构接地（1.5页，**当前整节标注"待重跑"**）
- **节首假设句（2026-07-01 评审意见）**："本节检验：把 LAFR 结构以对齐 token 注入冻结 LLM，是否让 LLM 真正*使用*了时序结构（判据：correct ≫ shuffled ≈ no_token），而非靠语言先验作答。"
- 数据：CARE Wind Farm C，16通道，窗口72，patch 6。
- 对比：`text_zeroshot` / `text_soft`（可训练软提示）/ `adapter`（ours），每种在 `correct`/`shuffled`/`no_token` 三条件下测，附 CoT 公平性检查（长思维链是否能让文本基线追上）与定性自由生成样例。
- **在写完整论文之前必须先跑通并保存**：`python experiments/llm_adapter.py --output-dir results/llm_adapter_full ...`、`python experiments/llm_compare.py --output-dir results/llm_compare ...`、`python experiments/lafr_eval.py --output-dir results/lafr_eval ...`（三者都已支持 `--output-dir`，只是从未在本工作区落盘过）。

### §5 Experiments II：AnomLLM 基准上的异常区间修正（1.5页，**已有真实数字**）
- **节首假设句（2026-07-01 评审意见）**："本节检验：让冻结 LLM 以结构化编辑操作修正 LAFR 候选区间，是否同时优于纯文本 LLM 基线与不接 LLM 的检测器本身。"
- 数据/协议：AnomLLM 官方合成生成器复刻（point/range/trend/freq），经典基线与纯检测器用全量 n=120/类，LLM 相关比较用 n=8/类（本地 Qwen3-8B 4bit 生成成本所限）。
- 指标：与 AnomLLM 完全一致的 affiliation precision/recall/F1（外加经典逐点 P/R/F1）。
- 主结果表 + trend 类特写（数字见 Claims-Evidence Matrix）。
- **必须在结果旁边同时讲清楚的三件事（否则审稿人会抓）**：① n=8/类样本小，需要扩大规模、跑多个 seed 报误差棒；② trend 类用了在训练集上校准的后处理策略（`calibrate_trend_policy`），要不要看起来像"偷看答案调参"取决于是否透明报告校准过程；③ 经典基线/纯检测器（n=120）和 LLM 相关方法（n=8）目前不是同一评测子集，正式结果表前需要在同一子集上重新算一遍纯检测器/经典基线做严格对照。
- 消融：`lag_readout_probe.py` 的 diff vs 互相关读出策略对比（**目前脚本只 print，需要改成落盘**）。

### §6 Discussion（0.5页 —— 原 1.0 页，2026-07-01 采纳评审意见压缩；诚实的局限 + 展望，**不是 Contributions**）
- 局限：n=8 样本量；trend 训练集校准的透明度；模型规模——已决策统一到 Qwen3.5-0.8B，若接口 B 最终保留 8B 则在此如实解释两种接入方式的规模敏感度差异。
- `forecast_head` 已有 Mackey-Glass 上的功能性验证（`results/mg_lafr_forecast/`，R²=0.982 vs naive −0.54），但对比对象只有 naive、数据只是合成单源序列——离"独立预测贡献"仍差一组真正的多步预测 benchmark（对比 PatchTST/iTransformer，用 ETT/Weather 等标准数据集）。写作时可作为附录 sanity check 引用，主文如果想主张"能预测"仍需诚实降级为 future work 或补齐 benchmark。
- **"给出控制系统的值"**：明确写成 outlook，不写成结果。概念上，同一套 event/pair token 只需换一个小回归头，或让 LLM 在编辑 JSON 里多输出一个"建议的 setpoint 调整量"字段，就能从"诊断"走向"建议"；但在没有真实控制回路或至少一个仿真器/代理任务之前，这只是方向。建议的第一步不是直接做闭环控制，而是一个轻量代理任务（例如：预测某个可控变量的"正常运行区间"，供人工/下游控制器参考），把它当作可评审的第一步而不是终局。
- 具体下一步实验：扩大 `--num` 并多 seed 出误差棒；补多步预测 benchmark；设计代理控制任务。

### §7 Conclusion（0.5页）
- 重申贡献（换措辞，不复制引言）、局限、1-2条具体未来方向。

---

## 3. Figure Plan

| ID | 类型 | 描述 | 数据来源 | 优先级 |
|----|------|------|----------|--------|
| Fig 1 | Hero / 架构图（建议拆成 2 个 panel，不要挤在一张里） | (a) LAFR 全流程 + 两种接入冻结LLM的路径；(b) 一个小 inset 对比"文本序列化"vs"token化"两种喂给 LLM 的输入形式 | 手绘 | HIGH |
| Fig 2 | 分组条形图 | CARE QA 上 correct/shuffled/no_token × {text_zeroshot, text_soft, adapter} 的准确率，证明接地是真的 | **待重跑** `llm_compare.py`/`llm_adapter.py` 输出 | HIGH（阻塞：需先重跑） |
| Fig 3 | 分组条形图 | AnomLLM 四类异常上 {llm, lafr_llm, lafr_llm_edit, lafr_detector} 的 affiliation-F1，trend 类单独高亮 | `anomllm-compare/lafr_qwen_token_policy_compare_results.json`、`lafr_qwen_trend_policy_results.json`（真实数据，可直接画） | HIGH |
| Table 1 | 主结果表 | 四类异常 × 5种方法（含经典 iso_forest/threshold）的 P/R/F1/affi-F1 | `compare_results.json` + `lafr_qwen_*.json`（**先解决 n=120 vs n=8 口径不一致**） | HIGH |
| Table 2 | 消融表 | lag 读出策略（diff vs xcorr，多个 softmax 温度）在 CARE 注入滞后对上的误差 | **待重跑并落盘** `lag_readout_probe.py` | MEDIUM |
| Fig 4 | 概念图（**必须标注"proposed, not evaluated"**；**2026-07-01 评审意见：移出正文，只放附录或直接删除**） | 从"诊断"到"预测 + 控制建议"的展望草图 | 手绘，仅用于附录 Discussion 补充 | LOW（附录） |
| Fig A1 | 线图（附录） | Mackey-Glass 上 forecast head 的预测 vs 真值 + 误差分布，作为 forecast head 功能性 sanity check | `results/mg_lafr_forecast/mg_lafr_forecast.png`、`mg_lafr_error_hist.png`（已有，可直接用或用 metrics.json 重画） | LOW（附录） |

**Fig 1 详细要求**：必须让"skim 读者"一眼看出两件事——① 同一个编码器可以用两种方式喂给 LLM；② 文本化会丢结构、token化不会（用一个小对比框直接画出"12, 15, 14, 22, ..."这样的文本 vs 一组带语义标签的 token 方块）。Caption 必须自洽：写清楚比较的是什么、读者应该注意什么。**箭头方向必须严格遵守 §0.5 的数据流定调：序列 → LAFR → 对齐产物 → 冻结 LLM → 答案，LLM 处于下游、无回边**——不能画成 LLM 调用时间模块的 agent 图。

---

## 4. Citation Plan（占位，**写作前必须逐条核实，不能凭记忆生成 BibTeX**）

- §1 Intro / §2(a) 文本序列化 LLM4TS：LLMTime (Gruver et al.) [VERIFY]、PromptCast (Xue & Salim) [VERIFY]、Time-LLM (Jin et al.) [VERIFY]、GPT4TS/One-Fits-All (Zhou et al.) [VERIFY]
- §2(b) 滞后/关系感知多变量模型：Crossformer (Zhang & Yan) [VERIFY]、iTransformer (Liu et al.) [VERIFY]、TimesNet (Wu et al.) [VERIFY]
- §2(c) 时序异常检测：Isolation Forest (Liu et al., 2008) [VERIFY]、Anomaly Transformer (Xu et al.) [VERIFY]、**AnomLLM 基准本身**（需确认准确题目/作者/venue，本仓库只留有其代码，未留论文引用信息）[VERIFY — 必须核实]
- §2(d) 冻结骨干+适配器：BLIP-2 (Li et al.) [VERIFY]、LLaVA (Liu et al.) [VERIFY]、Prefix-Tuning (Li & Liang) [VERIFY]
- §3 Method：affiliation-F1 指标的原始论文 [VERIFY]、CARE 数据集/基准论文 [VERIFY]
- §6 Discussion（outlook 部分）：LLM 用于预测标准基线 PatchTST / iTransformer [VERIFY]；LLM-for-control/decision-support 相关工作（如需要）[VERIFY]

**下一步建议**：正式写作前跑一次 `/citation-audit` 或 `/research-lit`，逐条核实作者/年份/venue，尤其是 AnomLLM 基准本身的准确引用信息——目前仓库里完全没有留下它的论文元数据，只有代码，这是最需要先查清楚的一条。

---

## 5. Reviewer Feedback（Step 6）

**✅ 外部交叉评审已完成（2026-07-01，GPT-5.5 xhigh via Codex MCP）**。评审结论："Based on the outline alone, I would not submit yet. The structure is coherent, but the strongest claims are still ahead of the saved evidence."

| 维度 | GPT-5.5 评分 | 最小修复（评审原话要点） | 是否已落进本计划 |
|---|---|---|---|
| 逻辑流 | 7/10 | §1 加一句"one mechanism, two validations"桥接句；§4/§5 开头各写明本节检验的确切假设 | ✅ 桥接句已有；§4/§5 假设句已补（见下） |
| Claim-证据对齐 | 4/10 | 把 C1/C2 移出贡献句、标为"pending rerun"；一句话贡献收窄到只主张已验证的 AnomLLM 结果 | ✅ 已在 §1 贡献列表加投稿版收窄规则 |
| 缺失实验 | 3/10 | 一张统一子集的合并结果表 + ≥3 seeds 的 mean±std；forecast head 不加 PatchTST/iTransformer 对比就不进正文 | ✅ 与既有 Next Steps 一致；forecast 已定为附录 |
| 与前人定位 | 5/10 | §2 加 4 行对比表（先前家族 / LLM 看到什么 / 滞后结构是否显式 / LAFR 多了什么）；替换全部 [VERIFY] | ✅ §2 已加表格要求；引用核实仍在 Next Steps |
| 页数可行性 | 4/10 | Mackey-Glass 只留附录；Fig 4 移出正文；§6 压到约半页 | ✅ 已执行：§6 → 0.5页，省出 0.5 页给 §3；Fig 4 改为附录 |
| Front-matter | 7/10 | 用 LAFR 命名的标题（候选2）；摘要开头给已验证的 AnomLLM 硬数字，并加一句 SCADA 线待重跑的 scope note | ✅ 已在 §0 摘要计划中标注 |

**2026-06-30 初版自查存档**（当时三条外部通道均不可用：codex 502 / oracle 缺 key / oracle browser 未登录，以下为按同样 6 维标准的自查，留作对照）：

| 维度 | 自评 | 问题与最小修复 |
|---|---|---|
| 逻辑流 | 7/10 | 两条实验线（问答接地 + 异常修正）如果不在引言里明确点出"同一机制两次验证"，容易读成两篇论文拼接。**修复**：已在 §1 做法概览里加入明确的定位句。 |
| Claim-证据对齐 | 8/10 | C1/C2 目前无保存结果却写进了 Contributions。**修复**：已标注"结果待重跑"，并在 §0 证据盘点里列为写作前必须完成的前置任务；若时间不够，应把 C1/C2 降级为"设计并展示协议，结果留待camera-ready/后续版本"而不是硬凑数字。 |
| 缺失实验 | 6/10 | n=8 样本量、无误差棒、trend 训练集校准透明度、n=120 vs n=8 口径不一致，四个问题都需要在 §5/§6 补实验或至少补充说明。已在对应章节标出。 |
| 与前人工作的定位 | 6/10（暂定，**引用未核实**） | 4 个子类划分合理，但所有引用都是 [VERIFY] 占位，AnomLLM 基准本身的准确出处未知。**修复**：写作前先跑一次文献核实。 |
| 页数可行性 | 6/10 | §3 Method 2页要装下编码器 5 个模块 + 2 种接口，偏紧。**修复**：已建议把详细张量公式移到附录，主文只留架构图+精简描述。 |
| Front-matter强度 | 7/10 | Fig 1 一开始想塞 3 件事（架构+两种接口+文本vs token对比），容易画乱。**修复**：已建议拆成 2 个 panel。 |

**关于"控制系统的值"的编辑判断**：目前的处理（完全从 Contributions/Experiments 里剔除，只放在 Discussion 当 outlook）是稳妥的默认选择——没有任何代码/数据/仿真支撑，硬塞进主张会是审稿人第一个打的靶子。如果你希望把它做实，建议的最小可行路径是：先做一个"预测某可控变量安全区间"的代理任务（不涉及真实执行器），跑出量化结果后再考虑要不要作为独立 Contribution——这值得先过一遍 `/experiment-plan` 单独设计，而不是直接在这版大纲里硬加一节。

---

## 6. 关于 venue 的建议

Skill 默认 `TARGET_VENUE = ICLR`（本次调用未传 `— venue:` 覆盖，故按默认值规划）。但这项工作同时有较强的工业应用背景（风电场 SCADA），如果你更看重"落地/决策支持"叙事而不是"通用 ML 方法"叙事，也可以考虑 AAAI/KDD 的应用track，或 IEEE Transactions on Industrial Informatics / Sustainable Energy 这类领域期刊（后者页数预算和引用格式都不同——引用计入总页数、用 `\cite{}` 数字引用而非 `natbib`）。这是可以随时用 `/paper-plan "..." — venue: XXX` 重新生成结构的一步，不影响已经做好的 Claims-Evidence Matrix 和证据盘点。

---

## Next Steps

**2026-07-01 范围决策（用户拍板）**：CARE 与 AnomLLM 两条重线暂停重跑，当前优先做**MG 简化验证线**——在 Mackey-Glass 合成数据上跑通"Qwen3.5-0.8B + LAFR + adapter 接地"核心机制（新脚本 [experiments/mg_llm_adapter.py](experiments/mg_llm_adapter.py)，输出 `results/mg_llm_adapter/`）。MG 线的优势：注入结构的 ground truth 由构造精确可知、无真实数据依赖、单卡笔记本可跑。Qwen3.5-0.8B 已确认可用（`Qwen/Qwen3.5-0.8B`，2026-03-02 发布，1.77GB，transformers 5.12.1 可加载，smoke test 通过）。下面 CARE/AnomLLM 各项保留为"恢复大论文计划时的待办"，不阻塞当前工作。

- [x] ~~第 0 步：确认 Qwen3.5-0.8B 的 HF 模型 ID/许可证~~ **已确认可用并 smoke 通过（2026-07-01）**；接口 B 的 JSON 编辑稳定性 smoke test 仍待做（仅当恢复 AnomLLM 线时需要）
- [ ] **先做**：重跑 CARE track 三个脚本（`lafr_eval.py` / `llm_adapter.py` / `llm_compare.py`），**骨干换 Qwen3.5-0.8B**，把 `--output-dir` 指向的 json 真正落盘；`lag_readout_probe.py` 加一个落盘出口再重跑
- [ ] 用同一 n（建议先把 LLM 相关比较的 `--num` 从 8 提到至少 30-50）重新跑一遍 AnomLLM track，**所有方法在同一子集上出一张合并结果表，≥3 seeds 报 mean±std**（2026-07-01 评审的最低要求）
- [ ] ~~统一或解释两条线的 Qwen3 规模差异~~ → 已决策统一到 Qwen3.5-0.8B（见 §0.5 架构决策；smoke test 失败则接口 B 保留 8B 并解释）
- [ ] 新增 C6"structure beats scale"三方对照：同一任务上 {0.8B+token, 0.8B+文本, 8B+文本}（与两条线重跑合并执行，不单独开跑）
- [ ] （可选，仅当想把"预测"写成贡献时）把 `mg_lafr_forecast.py` 的协议扩到标准预测基准（ETT/Weather），对比 PatchTST/iTransformer；否则 Mackey-Glass 结果只进附录
- [ ] 核实 AnomLLM 基准本身及其余 [VERIFY] 引用的准确信息（`/citation-audit` 或 `/research-lit`）
- [x] ~~待外部评审通道恢复后做真正的独立交叉评审~~ **已完成（2026-07-01，GPT-5.5 xhigh，6 条最小修复全部落进本计划，见 §5 Reviewer Feedback）**
- [ ] `/paper-figure` 生成 Figure 2/3 和 Table 1/2（需先有落盘的 json）
- [ ] `/paper-write` 起草 LaTeX（注意：正式论文正文按 output-language 协议应为英文，本计划文档按你的语言习惯保留中文）
- [ ] `/paper-compile` 编译 PDF
