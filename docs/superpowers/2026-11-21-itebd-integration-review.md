# InfiniteTEBD.jl 整合评审与默认策略

2026-11-21

保留按结构选择计算核的默认行为。会改变有限键维轨迹的选择，包括截断顺序、SWAP 路由和 Trotter 公式，由调用方决定。下一轮先统一演化、压缩和 gauge 恢复的契约。常用流程继续由 `evolve!` 和 `scarfinder!` 提供，底层调用可以逐步执行同一流程。 [上一轮实施与基准](plans/2026-11-21-itebd-next-optimizations.md)

## 评审发现

1. **Dense `canonical!` 的内存分派会改变有限键维结果。** 默认超过 1 MiB 时逐键投影已规范化态，然后重新恢复 gauge。小晶胞先截断 seam，再在已经截断的态上依次分解各键。对同一个固定种子的六站点随机态，$d=2,\chi=4$，直接比较两条路径，不截断时每晶胞归一化混合 transfer 重叠为 1，取 `maxdim=2` 时为 0.99645。两者的局部规范残差都在 $10^{-14}$ 以内。差异来自截断本身，检查规范残差无法发现。两站点 GHZ 在默认策略下发出简并警告，十五站点复制态走逐站点路径时通过且未发出对应警告。 [Dense 分派](../../src/iMPS.jl#L381) · [逐键投影](../../src/Schmidt.jl#L684)
2. **Gauge 恢复的调用时机取决于门类型。** 普通矩阵的 `recanonicalize=true` 只在 wrap gate 后生效，端点门则在任意位置对整胞再次调用带截断的 `canonical!`。对同一条跨第 2、3 站的非幺正对角门，前者留下约 1.1 的最大局部右等距残差。显式调用固定键维的 `restore_canonical!`，或用现有端点门路径恢复，例中的残差降至约 $10^{-14}$。两处现有布尔选项要按各自语义记录。若提供一次纯 gauge 恢复，先验收固定键维、态的保持和左右残差，再考虑通过高层传参。 [普通门](../../src/Gate.jl#L295) · [端点门](../../src/EndpointGate.jl#L186)
3. **ScarFinder 大晶胞压缩之后可能还未处于全局 canonical gauge。** `truncate_unitcell!` 的局部压缩可显式恢复 gauge，高层默认直接进入下一轮、能量校正或基于 Schmidt 谱的 refinement。八站点 $d=2$ 示例从 $\chi=6$ 压到 2，按默认扫掠得到约 $3.7\times10^{-2}$ 的局部右等距残差，整胞范数却已经接近 1。开启恢复后局部误差约 $10^{-15}$。因此只检查整胞范数会漏掉偏离。下一步要测恢复后的能量、最大键熵和选中的 refinement 时刻，并用归一化重叠区分 gauge 漂移与压缩误差。 [压缩](../../src/ScarFinder.jl#L161) · [ScarFinder 步骤](../../src/ScarFinder.jl#L488)
4. **TensorKit 的流式拆分已接入当前工作树，但接口仍有差异。** 目前 graded `applygate!` 只支持相邻两站点，`truncerr>0`、`mindim>1` 和 `return_stats=true` 会显式报错。左、右 transfer 的首特征值相合也不足以检验唯一性。发布验证要记录原版依赖组合，并优先补齐公开的精确截断诊断。 [TensorKit 规范化](../../ext/InfiniteTEBDTensorKitExt.jl#L953) · [TensorKit 门更新](../../ext/InfiniteTEBDTensorKitExt.jl#L1312)

第一项对照将 `MersenneTwister(533)` 生成的六个 `randn(rng, ComplexF64, 4, 2, 4)` 张量传给 `iMPS(ComplexF64, tensors; renormalize=false)`。Grouped 与 sitewise 在相同输入上使用 `cutoff=1e-12`。文中的重叠为每晶胞混合 transfer 谱半径除以两条自 transfer 谱半径的几何平均，右等距残差取所有站点的 Frobenius 范数最大值。

## 默认行为与用户选择

| 操作 | 当前可用的默认 | 整合建议 |
|---|---|---|
| `canonical!`，允许改键维 | 小 dense 晶胞走 grouped，大 dense 晶胞走逐键投影。TensorKit 用流式拆分。 | Dense 也实现保持原 seam、原切口顺序的流式 SVD，以 grouped 路径为小晶胞对照。有限键维下验证等价后，让内存分派只决定实现。若逐键投影在某些任务上有可量化的优势，再提供独立的压缩选项。 |
| `restore_canonical!`，固定键维并归一化 | Dense 和 TensorKit 默认 `:gram`。Dense 可选 `:qr` 和 `:auto`。 | Dense 的下一默认候选是已检查的 `:auto`，Gram 成功即返回，遇到已知数值失败再从原态运行 QR/LQ。保留显式 `:gram` 和 `:qr` 便于复现。只有在健康态开销、病态态成功率、近简并诊断及失败原子性验收后才改默认。TensorKit 继续使用 graded Gram。 |
| `applygate!` / `evolve!` 截断 | 完整 SVD。对角门与病态左 Schmidt 谱已有内部选择。 | 继续用完整 SVD 定义 `truncerr` 和 `BondStat`。QR-TEBD 的真实投影残差有独立含义，实验核继续用于量化可达到的精度和成本。 |
| `apply_endpoint_gate!` | `method=:mpo`，数值舍入阈值的算符 Schmidt 分解。 | 保留 MPO 默认。SWAP 路由会在搬运时引入额外的状态截断，继续用 `method=:swap` 显式选择。若满秩短区间确实需要直接融合，应先限制融合大小并证明其局部截断语义与 MPO 相合。放宽算符分解的舍入阈值须显式设置 `op_rtol`。 |
| Trotter 与 ScarFinder | 分层 `evolve!` 默认 `:second`。Hamiltonian ScarFinder 默认 `trotter=nothing`，保留旧顺序扫掠。 | 维持这两个默认值。三站点或多层实时间演化可选 `:fourth`，两层满足交换性的模型可选 `:fourth_opt`。比较不同阶数时固定物理时间和投影间隔，并计入局部门数与截断误差。 |

Dense 流式路线复用 TensorKit 和端点门已验证的数学原理。无截断的右向 LQ 令余下的物理区间成为行等距映射 $Q$。左向切口从 $MQ$ 缩到 $M$，且 $(MQ)(MQ)^\dagger=MM^\dagger$。要与旧 dense 路径一致，还要保留旧 seam 及其谱权重、每键选秩、归一化和截断后的恢复。固定点求解可在小键启动并传到既定 seam。截断开始的位置属于有限键维算法的一部分，不能随着最小键的位置改变。用可控简并谱验收临界切点，检查保留子空间与物理结果。 [已实现的流式分解](../../ext/InfiniteTEBDTensorKitExt.jl#L1072)

目前用公开的底层函数已经能执行一次定制 ScarFinder 更新，`gates` 可包含普通矩阵或准备好的端点门。下面假设压缩后的态在保留键上有完整数值支撑，并且具有唯一物理 transfer 固定点。

```julia
evolve!(ψ, gates, nstep; maxdim)
truncate_unitcell!(ψ, χ; recanonicalize=true)
# 如需约束能量，再用模型的 Hamiltonian 密度做校正。
# correction = fix_energy!(ψ, h, χ; target)
```

## 实施顺序

### 1. 统一规范化与压缩

- 先固化本页三个复现。Dense 流式路径应与 grouped 在同一 seam 上对照。测试包括不等键维、临界 `cutoff`、截断开关、缩放态和 seam 改变后的周期索引。显式定义 `renormalize=false` 在截断前后的整体缩放，历史 dense 和 TensorKit 的约定分别纳入回归。
- 让 `truncate_unitcell!` 从已经 canonical 的输入完成带上限的压缩，并给研究脚本保留单独的 `restore_canonical!`。现有门更新的 `BondStat` 来自局部切口。如果额外调用带截断的 `canonical!`，需要明确说明该步的额外截断。为 ScarFinder 验收压缩后的规范检查点，以实际能量和最大键熵决定默认恢复时机。能量修正也要在实际留下的态上验收。
- Dense 与 graded 的新诊断应覆盖 GHZ 型例子和近简并但仍 injective 的例子。局部等距和 Schmidt 环境残差检验 gauge，transfer 外围谱诊断另有用途。Dense `:auto` 如果两个求解都失败，应保留各自的失败原因。先限定支持的非 injective 情形，完整 multi-block canonical form 单独设计。

### 2. 优化 transfer 固定点和重复演化

- TensorKit 长晶胞基准中，32 站点固定点约 33.3 ms，流式拆分约 3.3 ms。先测最小 seam、`eigsolve` 的 matvec 次数和固定点残差，再决定何时试上次环境或快速局部检查。将候选传播回指定 seam 之后才做截断。上次环境只有在空间与基底兼容时才可重用。 [固定点基准](plans/2026-11-21-itebd-next-optimizations.md#阶段-d-实施与发布决定)
- Dense 复用 `ChainTransferWorkspace`，补上左向路径。Krylov 求解器保留的向量须拥有自己的内存。在同一次演化中量化 `chi_policy=:adaptive` 每次局部更新后调用全局 `canonical!` 的成本。逐门与每个完整 step 才恢复可能形成不同轨迹，先作为明确的调用策略对照。 [现有 transfer 工作区](../../src/TensorAlgebra.jl#L969) · [自适应调用](../../src/Gate.jl#L645)
- 长时间演化先复用已准备的门。一段 Trotter 演化可以按需生成调度，保持现有层融合的顺序和精确的测量、压缩时刻。在有限 `maxdim` 下两次半步中间多一次截断会改变状态，测试中要对照实际门序列。

### 3. 扩展结构化门与 ScarFinder 的公共组合方式

- 使 ScarFinder 的便捷入口可以接收与 `evolve!` 相同的已准备门序列。沿用现有名字与默认值，位置参数使用已有元组序列，允许包含 `EndpointGate`。有投影或远程门时，调用方可以继续提供独立能量算符。先补低层可组合性与准备成本的复用，再加这一个高层入口。 [现有 ScarFinder 组合](../../src/ScarFinder.jl#L506) · [门序列](../../src/Gate.jl#L797)
- 大 $d$ 优先实现由模型显式提供结构的精确门，例如 Hermitian 两体乘积先分别对角化两端，按 $O(d^2)$ 存储相位。三站点受控翻转可利用两侧投影子，只在允许的物理子空间作用。长程端点 Hamiltonian 项应能在两端指数化，随后走同一个准备门协议及逐站点期望值收缩。支持多站点 MPO 放在两个结构路径验证之后。
- TensorKit 先补精确的弃权、秩需求和 `return_stats`。用 Abelian 分块全谱及统一的全局秩规则，在可转 dense 的小例子上和相同 `truncerr` 对照。graded 长程门随后基于现有的守恒算符与传播接口扩展。

## 验收与发布

先用小系统精确参考校验 Trotter 和有无额外压缩的差异，再对 PXP、五态 clock 和守恒模型做端到端对照。PXP 固定投影间隔并跟踪回返、能量与最大键熵。Clock 扫 $dt,\chi$ 并跟踪多项观测量。长程门固定端点强度和间距，扫算符 Schmidt 秩，记录局部弃权和最终观测误差。性能报告按同等观测误差比较时间，分列门准备、transfer、施门、分解和诊断，单独记录累计分配与峰值内存。

上一轮 clock 例在 $dt=0.025,\chi=20$ 时，实验 QR 把演化从 411.6 ms 降到 265.4 ms，但最大键熵误差从 0.224 增至 0.239。另一组低算符秩端点门跨五站点，准备后 MPO 更新从 20.9 ms 降至 0.27 ms，报告精度内与直接稠密路径重叠为 1。这两组数据支持保留完整 SVD 默认，同时优先投入结构化门和等价的收缩优化。 [QR-TEBD 基准](plans/2026-11-21-itebd-next-optimizations.md#阶段-d-实施与发布决定) · [端点门基准](2026-11-21-itebd-performance-research.md#dense-实施与复测)

本轮还在三个健康的随机 dense 晶胞上比较了 `:auto` 和 `:gram` 的累计分配，排除输入复制，单线程 BLAS，预热后取九次中位数。对 $(n,d,\chi)=(4,2,4),(16,3,8),(32,4,16)$，`:auto` 的累计分配比 `:gram` 多约 3–6%。观测到同机运行负载波动，因此这组数据只用于估计分配开销，实际常用轨迹上的耗时仍需单独测量。

原版 TensorKit 0.17.1 在本机 Julia 1.12.6 触发 `_repartition_body` 的定义顺序错误。原版 TensorKit 0.17.3 已调整定义顺序。本轮用原版 0.17.3 在 Julia 1.12.6 上通过 1465 项，包括完整 symmetric 回归和 benchmark smoke。Legacy PXP 子进程第一次从仓库根目录启动时无法解析依赖，换到已解析完整依赖的临时目录后 4 项全部通过。文档构建因缺少独立的 Documenter 环境而跳过。Julia 1.10 和 TensorKit 0.16 仍需无本地补丁的发布验证。 [TensorKit 0.17.3](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.17.3)

## 整合结果

- **Dense canonicalization。** 大晶胞和异质物理空间用流式拆分。固定点从最小虚键求解并传播回原 seam，沿用原 seam 的谱权重、逐键截断顺序和必要的第二次规范化。小晶胞保留 grouped 对照。六站点 $d=4,\chi=12$ 的随机态在 cap 8 下比较单次 seam 与分解，归一化混合 transfer 重叠为 1，逐键 Schmidt 谱最大差约 $1.4\times10^{-15}$。单线程实测 627.6 → 3.23 ms，累计分配 1159.94 → 1.35 MiB。相同局部维数和 cap 下，流式路径从 12 到 24 站点的累计分配为 2.12 → 3.59 MiB。大虚键的非 injective 检查仍属于有限诊断。 [流式算法](../../docs/src/imps.md)
- **Gauge 与 ScarFinder。** `applygate!` 可用 `gauge=:auto` 等方法在单门后做纯 gauge 恢复。相同关键词在 `evolve!` 的一次 sweep 或一个 Trotter macro-step 后生效。TensorKit 使用 `:gram`。高层 ScarFinder 默认修复大晶胞压缩后以及 energy-fix 测量前的 gauge。显式 `recanonicalize=false` 可继续使用原流程。独立的预制 gate 序列可以驱动主演化和 refinement。八站点和十二站点 PXP，在 $dt=0.04,\chi=2$、每轮 5 个 gate sweeps、主演化与反向 refinement 各 5 轮的对照中，局部右等距最大残差由 $10^{-5}$ 降至 $10^{-15}$，独立重新规范后计算的能量差从约 $10^{-7}$ 降至约 $10^{-9}$，两条流程均选中第 5 个 refinement 样本。累计分配增加约 10–12 MiB。 [ScarFinder 工作流](../../docs/src/scarfinder.md)
- **Transfer 与门计划。** Dense 左右 transfer 共用可复用工作区，每次返回 Krylov 自己持有的矩阵。对 $n=32,d=4,\chi=16$ 单次左向 matvec，与这轮改动前的分配版本相比，累计分配约从 1.14 MiB 降到 0.007 MiB。TensorKit 的纯 gauge 路径可从最小虚键求解，环境候选通过残差检查才复用。分层 `evolve!` 复用重复的 Trotter 序列和融合边界。两层四阶、10,000 步的准备分配从 43.86 MiB 降至约 0.013 MiB，逐门序列与公开 `trotter_gates` 一致。若显式请求逐 macro-step gauge 检查点，边界的两个半步各自保留。 [演化选项](../../docs/src/time-evolution.md)
- **结构化门与精确统计。** `EndpointGate(A,B,scale)` 从两端尺寸的相位表准备乘积项。三站点 `ControlledGate(P,U,Q)` 利用 $I+P\otimes(U-I)\otimes Q$，直接走局部传递与同序 SVD。对 $d=8$、输入 $\chi=6$、cap 8 的单门，与完整稠密门归一化重叠为 1，更新约 8.64 → 0.21 ms，累计分配 6.58 → 0.33 MiB。TensorKit 的 Abelian 两站点路径增加 `mindim`、`truncerr` 和 `return_stats`，按所有 sector 的统一谱选择秩。Dense 和 graded 的弃权计算都直接累计小奇异值的权重，避免用 1 减去接近 1 的保留比例。 [TensorKit 支持范围](../../docs/src/symmetries.md)

这轮实现中，Dense 固定维数的 `restore_canonical!` 使用 Gram。病态基底可显式选择 `:auto` 或 `:qr`。完整 SVD、端点 MPO 和分层二阶 Trotter 继续是标准路径。基准为预热后单线程局部计时，累计分配量与峰值 RSS 分开解释。

在重复已 canonical 的 32 站点 U(1) 晶胞上，TensorKit 的残差检测确认已有环境可用，固定点阶段相对于总是启动 Krylov 从 34.63 降到 2.01 ms，累计分配从 25.10 降到 1.45 MiB。非平衡态或每步大幅改变的态需要实际 Krylov 求解时，不能直接套用这项提速。 [基准入口](../../bench/bench_symmetric_qr.jl)

Julia 1.12.6 搭配原版 TensorKit 0.17.3 的整合套件通过 1657 项，包括 AKLT、对称 Heisenberg、benchmark 和 legacy PXP。最后更新的 dense 门、演化和截断专项通过 306 项，逐项复测了失败的 gauge 检查点和零权重的 `mindim`。Julia 1.10.13 搭配原版 TensorKit 0.17.3，通过 445 项对称性回归和 139 项 ScarFinder 回归。Documenter 完整构建通过。原版 TensorKit 0.16.5 在 Julia 1.10.13 上也复现了上游 `_repartition_body` 的方法定义顺序错误，因此文档环境使用 0.17.3。 [Julia 发布版本](https://julialang.org/downloads/) · [TensorKit 发布版本](https://github.com/QuantumKitHub/TensorKit.jl/releases/tag/v0.17.3)
