# Research Output Manifest

> Auto-maintained by ARIS skills. Tracks all generated artifacts across the research lifecycle.

| Timestamp | Skill | File | Stage | Description |
|-----------|-------|------|-------|-------------|
| 2026-06-30 21:18 | /paper-plan | PAPER_PLAN_20260630_211834.md | paper | LAFR+冻结LLM论文大纲初版：token接地+证据到编辑两条机制，AnomLLM基准有真实数字，CARE track结果待重跑，控制系统值仅作outlook |
| 2026-06-30 21:18 | /paper-plan | PAPER_PLAN.md | paper | latest copy |
| 2026-07-01 17:24 | /paper-plan | PAPER_PLAN_20260701_172437.md | paper | 增量更新：新增§0.5架构数据流定调（序列→LAFR→对齐token→冻结LLM，非LLM调用时间模块）；C4升级为部分支持（纳入results/mg_lafr_forecast/ Mackey-Glass落盘结果）；新增Fig A1附录图 |
| 2026-07-01 17:24 | /paper-plan | PAPER_PLAN.md | paper | latest copy |
| 2026-07-01 18:25 | /paper-plan | PAPER_PLAN_20260701_182526.md | paper | 架构决策落地：核心方案定为"统一入口+内部路由"（非agent式），冻结骨干统一为Qwen3.5-0.8B（接口B先smoke test，退路8B+解释），新增C6 structure-beats-scale三方对照，Next Steps增加第0步 |
| 2026-07-01 18:25 | /paper-plan | PAPER_PLAN.md | paper | latest copy |
| 2026-07-01 22:09 | manual | PAPER_PLAN_20260701_220901.md | paper | MG简化验证线正式结果落地：新增证据盘点行+C7，总体接地成立(gap+0.256)，2/3任务grounded（conf+0.542/lead+0.215），cp任务异常(gap-0.107)需诊断；Next Steps新增cp诊断+多seed待办 |
| 2026-07-01 22:09 | manual | PAPER_PLAN.md | paper | latest copy |
| 2026-07-01 22:09 | manual | mg_llm_track/README.md | docs | 回填正式运行结果（耗时12679s），含分任务表格与cp异常的三个候选解释 |
| 2026-07-01 23:14 | manual | experiments/mg_llm_explain.py | code | 解释版MG实验：训练目标改为完整解释句（由注入GT自动构造），LLM自由生成+事实核查评估，保留三重接地对照；含--demo交互模式 |
| 2026-07-01 23:14 | manual | PAPER_PLAN_20260701_231457.md | paper | C7扩展为跨协议证据：解释版conf 0.800/lead 0.704接地（gap+0.229/+0.296），cp恒答Yes不接地，跨两协议复现 |
| 2026-07-01 23:14 | manual | PAPER_PLAN.md | paper | latest copy |
| 2026-07-01 | manual | experiments/mg_llm_adapter.py | code | MG简化验证线：MG序列→LAFR(冻结)→adapter对齐token→冻结Qwen3.5-0.8B→答案，复用llm_adapter.py的三重接地协议(correct/shuffled/no_token) |
| 2026-07-01 | manual | mg_llm_track/README.md | docs | MG LLM track独立文档文件夹：说明架构、依赖、运行命令、结果解读标准；不复制脚本，只引用experiments/下的共享模块 |
