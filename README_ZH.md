# SETSUNET 核心精简版

本仓库提供 SETSUNET-CSN 的核心 Python/C 源码、唯一共享的 **49,096 参数模型**，以及不依赖大型数据下载的精简测试。示例固定 **五个外层步骤**，所有案例使用同一套权重。

## 运行

需要 Python 3.11+ 和 C 编译器。建议先创建并激活虚拟环境，在仓库根目录执行：

```sh
python -m pip install -r requirements.txt
python scripts/build_native.py --cc cc
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/run_cases.py
```

Windows 本轮验证使用 TinyCC 0.9.27（x86-64）；将下面路径替换为实际位置，等待编译完成后再测试：

```sh
python scripts/build_native.py --cc C:/path/to/tcc.exe --tcc
```

也支持 `--cc C:/path/to/zig.exe --zig`。仅需 CPU PyTorch。示例输出位于 `results/compact.json`，逐例列出 CSN、同曲率直接 LM 控制和 PCG16 的五步结果。

## 内容

- `code/setsunet_csn/`：共享网络、测量子空间、Schur 补全、双视图 minimax、白化和五步求解器；保留必要的底层依赖模块及可微轨迹实现。
- `code/native_stage4/`、`code/native_stage5/`：两个 C 内核源码；不包含编译器与预编译二进制。
- `code/checkpoint/`：约 371 KiB 的单共享权重，NPZ 格式，附架构和逐张量哈希。
- `tests/`：60 个 SPD 数值性质检查、12 个 C/参考 minimax 对照、秩亏正交化检查；四类生成案例的 12 条五步路线；四个历史曲线包络参考终点。
- `scripts/`：编译、精简示例和发布文件校验入口。

主要入口是 `setsunet_csn.stage5.solve`，加载权重使用 `compact_runtime.load_model()`；顶层历史 `setsunet_csn.solve` 对应早期实现。曲线包络是可选研究模块，默认示例使用原正式直线求解器。

测试案例包括 96 维平衡/噪声/偏置双视图图标定和 64 维 Broyden 方程，数据由固定种子生成。四个历史参考终点保留原案例 ID 和实际开发/确认划分，用于检查导出等价性。这些软件测试与论文的“21 项异构任务基准”分别命名，不替代原逐任务实验结果。

## 分析报告与基准结果

[证据目录](evidence/README.md)收录最新论文和附录、九张论文表的 CSV、原始 21 项异构任务的后续逐任务实验、求解器分析，以及后来的 126 例外部方法比较和修复版复核。论文保留的 18 项任务与原始 21 项任务分别标注；126 例研发版本的结果不等同于此处默认推理入口。

可下载完整的[证据压缩包](SET-Work-evidence-20261002.zip)。[证据清单](evidence/MANIFEST.json)记录来源与 SHA-256；`python scripts/check_evidence.py` 核对目录及压缩包。核心代码压缩包与根目录清单对应精简代码发布。

未包含大型图像数据、完整历史运行归档、运行环境、第三方模型和被拒绝的训练候选。依赖外部数据与旧接口的完整实验/训练编排不属于本精简包；遗留 `from_profile` 接口需要原配置文件。原模型和原实验目录未被修改。

来源及哈希见 `SOURCE_PROVENANCE.json`、`MANIFEST.json`；环境与验证见 [VALIDATION.md](VALIDATION.md)。本轮本地验证平台为 Windows，Linux/macOS 编译分支尚需对应平台验证。许可范围见 [LICENSE_NOTICE.md](LICENSE_NOTICE.md)。
