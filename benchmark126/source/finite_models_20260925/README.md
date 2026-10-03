# SETSUNET 两个有限模型完成版

两个入口均共享原来的49,096参数检查点，执行五个外层步骤，第2步进行一次有限模型修正。当前发布是经过44个既有案例核验的研究实现，未替换项目正式策略。没有新增训练，也没有按任务切换权重或候选。

| 发布名称 | 入口对应实验名 | 本轮采用的改变 |
|---|---|---|
| `path_line_complete` | `path_line_complete` | 显式求解入口、紧凑低秩非线性评价、状态缓存、有限候选检查、数值失败回退；保留原路径及线性问题的有限迭代行为 |
| `quadratic_usable8_complete` | `quadratic_polished8` | 保留原8维有效方向和二次拟合；给原候选增加三轮精确坐标四次多项式求解，并保留原候选参与真实观测目标验收 |

## 调用契约

从项目根目录使用 `ao311` 环境，将 `tools` 与项目原来的 `workspace/scripts`、`workspace` 加入Python导入路径。现有runner完成这些配置。公开入口是 `finite_models_20260925.release.solve`：

```python
from finite_models_20260925.release import solve

result, packet = solve(
    problem=prepared_problem,
    network=shared_network,
    engine=counted_engine,
    outer_config=five_step_config,
    model="quadratic_usable8_complete",  # 或 path_line_complete
)
theta = result["theta"]
counts = result["physical_counts"]
failures = result["refinement_failures"]
```

这里的 `prepared_problem`、`counted_engine`、`five_step_config` 使用项目已有的原生残差适配器、CountedEngine和Stage5Config。求解输入不包含评分器、真值、case名称或任务类别；任务原有外层配置由调用方保留。CPU float64代数，原项目PyTorch网络和C内核继续使用，新模块以NumPy/SciPy实现。返回诊断包包含拟合查询、候选状态、选择依据和数值状态，便于独立重评分。

`problem.native.least_squares(x, jacobian)` 必须返回与 `0.5*r@r` 对应的双视图平均观测代价；接口检查会核对 `native.cost(x, view)`。缓存和临时计数回调在 `finally` 中恢复。秩亏、非有限值和线性代数异常记录原因并退回该步原曲线提议。非数值编程异常不伪装成成功。SLSQP非零状态可留下可行有限候选，但明确记录，不能称作全局收敛。

## Path-Line Complete

原网络先提出更新。在原提议点附近，按Jacobian观测依赖结构恢复全维二次残差模型。通过GN锚点处的两个输出模式压缩，求一个全维候选，然后在GN与候选之间的线段上解四次目标。原始步、半步和四分之一步经真实观测目标验收。

本轮把非线性锚点模型评价改写为小输出模式加预计算二次型，避免搜索中反复处理完整残差输出；状态变量仍为全维。用32个随机模型验证其目标和梯度与稠密表达一致。对恢复出的二次响应恰为零的情形，保留旧有限迭代求解路径：精确凸球求解虽然数学上更彻底，却在两个偏置案例改变了极小量的恢复误差，因此作为消融留档，不自动采用。

当前44例五步结果全部与旧 `path_line` 在论文数值容差内打平，最大绝对误差差约5.7e-13。本轮属于工程和数值实现完成，**不声称路径模型出现新的恢复精度突破**。实测总joint J为13–21。仍保留中心支撑假设和局部秩亏诊断；没有将有限探针当成完整物理Hessian证书。

## Quadratic Usable8 Complete

原提议位移、投影后网络方向及补足的测量方向构成最多8维局部坐标。保留原采样节点和完整二次残差Hermite拟合，原16起点SLSQP也保留。每个原入选候选增加三轮坐标优化：固定其他坐标后，残差沿一个坐标是二次函数，目标为四次函数；枚举导数实根与边界得到该坐标可行区间内的最优值。

可行域继续为 `|z_i|<=1` 与 `||z||<=2` 的交集。坐标上界从剩余球半径计算，每次坐标更新都核验模型代价不增。保留原候选，再额外实际验证去重后的改进候选。该阶段不增加Jacobian查询，最多增加三个残差前向候选。实测总joint J仍为13–18（旧版为14–18；缓存可复用部分已测状态）。

44个原二次拟合包的中心、基底、节点、设计矩阵、系数和QR压缩均做逐项兼容检查。Rat43组均值从0.2378123降到0.1942028；改善集中在一个案例。TUM和SE3各一例略退步，详细数值见发布报告。不能把观测目标下降表述为最终真值误差保证。

## 保留的未采用路线

`api.py` 和 `ModelConfig` 保留第一轮四种研究配置，`search_completion.py` 还保留三角区域搜索，`path_radial.py` 保留真实径向拟合。这些不是默认发布选择。

- 中心残差/J强制精确：部分光度/TUM收益，SE3退步。
- 用已观测线性步校准二次项幅度：SE3退步进一步扩大。
- GN—锚点真实Hermite拟合：改善一例Rat43，但部分光度恢复退步。
- 三角区域与实际径向路径扩展：没有稳定恢复收益。

历史源码、协议、失败路径和所有端点均保留；不通过真值选择每个case的最佳路线。

## 验证与结果

四轮共352个新端点，只使用用户指定的同一44例；模型研发与评估复用这些案例，不能称作未见数据泛化。四轮均在独立进程中重评分并检查候选目标、缓存数据与第2步状态。代数检查覆盖32个紧凑模型、32个三角模型、320个三角域保范点、二次模型未拟合点、平坦多项式和可行域投影。

发布检查：`tools/check_two_model_release_20260925.py`。它还对CG2 Broyden与CG0 SE3做原控制器逐参数兼容和两种模型的强制失败回退检查。首次检查的Python绑定方法对象身份断言修正在记录中保留。

报告与表格：`workspace/results_stage9_dev/runs/two_model_release_20260925/`。正式论文、原权重和active policy保持。研究完成度包含可调用接口、可恢复实验、独立评分、完整负结果和故障处理；尚不包含全任务泛化证明、C/C++移植或正式速度基准。
