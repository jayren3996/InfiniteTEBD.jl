# InfiniteTEBD.jl 后续优化实施方案

2026-11-21

先做两项可以与当前实现逐步对照的改动。TensorKit 用局部 LQ 加原顺序 SVD 取代整胞拆分。Dense 两站点核先保留结构化门，再改善含很小 Schmidt 值的张量重建。近奇异 gauge 恢复和改变截断子空间的 QR-TEBD 分别验收。现有公共入口负责常用流程，研究脚本仍可以直接组合底层操作。 [现有研究和基准](../2026-11-21-itebd-performance-research.md)

## 1. TensorKit 先等价替换整胞拆分

TensorKit 已经用逐站点 transfer matvec 求左右固定点。指数增长出现在 `_split_unit_cell!`，它把经过 seam gauge 处理的全部张量合并，再从左往右截断。第一阶段保留跨缝 gauge、边界权重、截断顺序及范数约定。现有 TensorKit `renormalize=false` 仍会归一化物理晶胞，并保留自然 Schmidt 缩放。这个行为应先写成回归测试。 [当前实现](../../../ext/InfiniteTEBDTensorKitExt.jl#L805)

1. 固定现有 seam 处理结果。从第 \(n\) 个站点到第 2 个站点做无截断的块稀疏 LQ 或 `right_orth`。用 TensorKit 的 repartition 处理对偶腿和融合树。把每步左因子传给前一站点，最后一个左因子完整吸收进已经带 `Λwrap` 的第一站点。
2. 从左往右逐键用原有 `svd_trunc!`，沿用 `truncrank(maxdim) & trunctol(atol=cutoff)` 及谱归一化。原切口矩阵可以写成 \(MQ\)，其中后缀 \(Q\) 是行等距矩阵。所以 \((MQ)(MQ)^\dagger=MM^\dagger\)，用较小的 \(M\) 就能得到同一条切口的非零奇异谱。若阈值切开简并谱，比较物理量和保留子空间，而非逐个特征向量。
3. 保留小晶胞整块路径作为参考。分派时使用单融合张量的保守字节估计，避免为估算先构造完整 graded ProductSpace。候选张量与谱通过检查后再写回。以固定键维估算，局部扫掠的存储随晶胞长度线性增长。 [TensorKit 分解](https://quantumkithub.github.io/TensorKit.jl/stable/man/factorizations/) · [MPSKit 正交扫掠](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/src/states/ortho.jl#L201-L328)

之后再补 TensorKit 的 `restore_canonical!` 方法。纯 gauge 恢复应保持键空间和电荷标签，重叠对照以**已经截断的态**为基准。TensorKit 的左 transfer 是物理伴随，dense `dir=:l` 按本库向量化约定作用为转置，需要在复数测试里区分。左右 Perron 根相符不足以排除多个独立物理固定点，固定点残差、PSD 性和 sector 策略分别验收。 [TensorKit transfer](../../../ext/InfiniteTEBDTensorKitExt.jl#L592) · [Dense transfer 约定](../../src/observables.md#matrix-free-transfer-maps)

**验收。** 同时跑 TensorKit 0.16 和 0.17。用 U(1)、Z₂ 和复合 Abelian 对称性覆盖不同键空间、物理电荷、紧 `maxdim` 和临界 `cutoff`。看逐键分 sector 的 Schmidt 权重、左右规范残差、归一化态重叠、局部观测量和弃权。谱权重用 TensorKit 的范数。小态可以转 dense 做 oracle，重叠也可用测试内部的 graded 混合 transfer。大晶胞测 \(n=8,16,32\) 的峰值内存，分开记录固定点和拆分耗时。增加失败时原态保持完整的测试。

## 2. 近奇异环境用周期 QR/LQ 恢复 gauge

当前 dense `restore_canonical!` 由 transfer Gram 开方和求逆得到 gauge。小环境特征值会放大误差，物理态本身正常时也可能因为虚键基底病态而触发。参考 TeNPy 的 `canonical_form_infinite2` 和 MPSKit 的迭代正交化。ITensorInfiniteMPS 可用于参考局部 polar sweep，它的初始环境仍来自 Gram。只扫一圈并丢掉最后的环闭合因子会改变物理态。 [本库 Gram 路径](../../../src/Schmidt.jl#L372) · [Vanderstraeten 等，`2.3](https://arxiv.org/pdf/1810.07006) · [TeNPy](https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mps.py#L4366-L4497) · [MPSKit](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/src/states/ortho.jl#L232-L328) · [ITensorInfiniteMPS](https://github.com/ITensor/ITensorInfiniteMPS.jl/blob/main/src/orthogonalize.jl#L4-L122)

1. 在较小的虚键初始化 seam carry。从右向左做局部 LQ，写为 \(A_iR_i=a_iR_{i-1}B_i\)。固定相位约定，累计尺度，直到归一化 seam carry 闭合。先验证纯扫掠；transfer 间隙小导致收敛慢时，再用 mixed-transfer Arnoldi 加速。
2. 在右等距 \(B_i\) 上从左向右做 QR，求满足 \(C_{i-1}B_i=A_{L,i}C_i\) 的中心因子。检查中心方程与周期闭合，只对虚键中心做 SVD \(C_i=U_i\Lambda_iV_i^\dagger\)。存储 \(V_{i-1}^\dagger B_iV_i\) 与各键的 \(\Lambda_i\)，需要时由累计的对数尺度还原输入范数。
3. 先用显式 `restore_canonical!(ψ; method=:qr, tol, maxiter)` 验证，再评估自动回退。回退只处理已经识别的数值问题，例如 Gram 近奇异、局部残差过大或固定点失败，并从原输入重试。固定键维操作遇到真正空支撑或不唯一的固定点应报告，秩压缩留给显式截断。

**验收。** 用已知 injective 复数态构造固定物理态、虚键条件数从 \(10^8\) 到 \(10^{10}\) 的例子。加入不等键维、绕回晶胞、Float32、真秩亏和近多固定点。共同检查周期闭合、左右规范条件、归一化态重叠和局部观测量。仅靠右等距残差无法证明物理态保持。公布同一精度下的迭代次数和耗时，再决定是否开启自动回退。

## 3. 大物理维数先做精确两站点核

分开测门准备、门收缩、分解和诊断。一般稠密两站点门收缩约为 \(O(d^4\chi^2)\)，全 SVD 约为 \(O(d^3\chi^3)\)。现有 `Diagonal` 更新只乘相位，但 `trotter_gates` 的矩阵缓存会稠密化门。先让全对角门列表保留结构。Hermitian 乘积项 \(h=A\otimes B\) 可以分别对角化两端，存 \(d^2\) 个相位并用局部基变换施门。在模型显式提供该结构时复用这一路径。 [本库门缓存](../../../src/Gate.jl#L516) · [Diagonal 更新](../../../src/TensorAlgebra.jl#L128)

**无逆重建。** 先构造尚未乘左 Schmidt 谱的已施门块 \(C\)。对 \(M=\Lambda_LC\) 用现有完整 SVD 和同样的选秩规则，右张量取保留右奇异向量的伴随，左张量用 \(CV_k\) 重建。归一化时除以原保留奇异值的范数。这样不会显式除以很小的 \(\Lambda_L\)，仍可返回现有 `BondStat`。测试精确零权重行和很小非零权重行。该重建增加一次矩阵乘法，稳定性和速度分别衡量。 [TeNPy 两站点核](https://github.com/tenpy/tenpy/blob/main/tenpy/algorithms/tebd.py#L408-L442) · [本库截断规则](../../../src/TensorAlgebra.jl#L457)

**QR-TEBD 和 controlled bond expansion。** 用旧右等距空间与正交残差方向构造维数 \(\eta\) 的扩张子空间，再做低维 QR/LQ。当 \(\eta=O(\chi)\) 时，截断阶段可降到 \(O(d^2\chi^3)\)，一般稠密门的收缩成本另算。首版放在私有实验核，限定相邻 dense 实时间幺正门。 [Unfried–Hauschild–Pollmann，`II–III](https://arxiv.org/html/2212.09782) · [TeNPy QR-TEBD](https://github.com/tenpy/tenpy/blob/main/tenpy/algorithms/tebd.py#L574-L682)

对于 QR 提议的列等距右空间 \(Q_R\)，写回前重算 \(M Q_R\)，在这个小核做薄 SVD 并旋转右空间。对最终保留空间 \(V_k\) 测 \(\|M-(MV_k)V_k^\dagger\|_F^2/\|M\|_F^2\)。残差接近舍入误差时直接算残差，避免用两个接近的范数相减。这个量可以认证候选，但无法恢复完整谱的最小 `chi_req` 或最大被丢弃奇异值。首版遇到 `return_stats=true` 或正 `truncerr` 仍用完整 SVD。QR 候选采用单独的投影误差门槛，必要时扩大子空间或退回完整 SVD。 [TeNPy QR 分解与误差](https://github.com/tenpy/tenpy/blob/main/tenpy/linalg/truncation.py#L589-L654)

**验收。** 用 \(d=2,4,8,16\) 测不同门型和键维，加 \(d=5\) clock quench 覆盖快速纠缠增长。以完整 SVD 为对照，在匹配局部残差和长时观测误差下报告速度、峰值内存、扩张次数和回退比例。固定物理时间和 ScarFinder 投影间隔，分别扫 \(dt\)、\(\chi\) 和 QR 容差。

## 发布顺序

| 阶段 | 改动 | 进入下一阶段的条件 |
|---|---|---|
| A，可并行 | TensorKit 局部拆分；dense 保留对角门、无逆两站点重建 | 小晶胞对照通过，长晶胞无需融合物理张量，精确统计兼容 |
| B | 显式 dense 周期 QR/LQ | 病态案例恢复原物理态，闭合与规范残差达标，失败可安全重试 |
| C，可并行 | TensorKit 纯 gauge 恢复；dense QR-TEBD 实验 | graded 谱与空间正确；在匹配误差下测到 QR-TEBD 收益 |
| D | 评估自动分派、TensorKit 版 QR-TEBD | 有独立的收敛和性能数据，新增诊断与截断契约清楚 |

## 阶段 A 实施记录

TensorKit 的生产路径已使用逐站点块稀疏 LQ 与同序 SVD。旧整块拆分仅作为小晶胞对照。公开 `canonical!` 在候选态上完成分解，再通过键空间和 Schmidt 谱校验后提交。新增 U(1)、Z₂、U(1)×Z₂ 回归，覆盖不同键空间、截断阈值、归一化选项、32 站点晶胞和失败原子性。 [流式拆分](../../../ext/InfiniteTEBDTensorKitExt.jl#L920) · [分 sector 对照](../../../test/test_symmetric_basic.jl#L19)

Dense 门准备保留 `Diagonal` 类型，含稠密门的混合序列也保留对角门的结构。相邻 dense 核用最小非零入射 Schmidt 值与最大值之比判断是否需要无逆重建，阈值为输入实精度的 \(\sqrt{\epsilon}\)。条件良好的谱继续用原核。两条路径共用选秩和 `BondStat`。单线程 \(d=16\) 门准备微基准中，对角存储约 0.007 ms、0.009 MiB，等价稠密存储约 0.329 ms、2.002 MiB。\(d=8,\chi=16\) 配对实验中，强制无逆重建约 13.3 ms，原核约 10.4 ms，这项代价支持只在需要时开启。累计分配量不等于峰值内存。 [Dense 更新](../../../src/Gate.jl#L55)

本机 Julia 1.12.6 在原版 TensorKit 0.16.5 和 0.17.1 的既有分块操作中触发 `_repartition_body` 的定义顺序问题。分别用临时 TensorKit 副本将该辅助函数移到生成函数之前，两个版本的对称性测试均通过 379 项。仓库的依赖及版本约束未改。32 站点功能测试已通过，峰值内存曲线仍待系统测量。

## 阶段 B 实施记录

`restore_canonical!(ψ; method=:qr)` 已支持 dense iMPS。周期迭代在相对 seam 残差达标后才生成右等距张量，再用左扫掠恢复逐键中心谱。检查 Schmidt 环境关系后一次性写回。算法保持键维，归一化每晶胞，拒绝数值秩亏或迭代次数耗尽的结果。小 seam 额外检查多重外围本征值。默认 `:gram` 和 `canonical!` 分派保留。 [代码](../../../src/Schmidt.jl#L356) · [规范化回归](../../../test/test_imps_canonical_api.jl)

同一物理态上施加条件数 \(10^8\) 至 \(10^{10}\) 的复数虚键变换，QR/LQ 恢复了逐键 Schmidt 谱、局部观测量和归一化重叠。一个独立微基准里，Gram 在 \(10^8\) 和 \(10^{10}\) 下报告 transfer 环境错误，QR 的右等距残差约 \(10^{-15}\)。小谱隙回归确认有限扫掠预算可耗尽，输入保持完整。单线程 Julia 1.12.6、七次中位数、不含输入复制的随机态微基准如下。

| 晶胞、局域维数、键维 | Gram 时间 / 分配 | QR 时间 / 分配 |
|---|---:|---:|
| \(n=8,d=3,\chi=4\) | 3.71 ms / 1.02 MiB | 0.37 ms / 0.80 MiB |
| \(n=16,d=3,\chi=8\) | 5.03 ms / 6.82 MiB | 0.93 ms / 2.18 MiB |

两个健康态对照的归一化混合 transfer 重叠在所报精度下均为 1。本轮未将 QR 接入自动回退，mixed-transfer 加速及 QR-TEBD 截断属于独立改动。

## 阶段 C 实施记录

TensorKit 的 `restore_canonical!(ψ)` 已公开固定键空间的 Gram 恢复。它从最小 graded 键开始，逐站点传播两个环境，对每个虚键的方阵做分块分解，并分别验证右等距条件和左 Schmidt 环境。验证失败时原态保持完整。它归一化每个晶胞，并要求已有键空间具有完整数值支撑。需要截断可调用 `canonical!`。复数 U(1)、Z₂ 和 U(1)×Z₂ 均有空间与混合 transfer 重叠对照，另有 32 站点和原子失败测试。 [实现](../../../ext/InfiniteTEBDTensorKitExt.jl) · [TensorKit 文档](../../src/symmetries.md)

Dense QR-TEBD 仍是私有实验核。它从上一轮右等距空间出发，用投影遗漏权重较大的行扩张，做 QR/LQ 后重新分解实际右投影的小核。返回候选误差、是否达到目标，以及是否使用 QR。若候选误差超阈值，退回原全 SVD。若全 SVD 在当前键维上限下仍达不到目标，返回该键维下的最优结果并标记未达标。常用 `applygate!`、`truncerr` 和 `BondStat` 继续使用原路径。 [实验核](../../../src/ExperimentalQRTEBD.jl)

单线程 Julia 1.12.6 的近低秩合成切口对照，输入维数为 \((d\chi)\times(d\chi)\)，固定 \(\chi=12\)。种子 6023，矩阵是高斯左因子乘正交右基的伴随，加上系数 \(10^{-4}\) 的独立高斯扰动。两法保留相同的秩。表中的时间为七次中位数，只含切口分解与候选误差检查，排除了门收缩。

| \(d\) | 完整 SVD | QR 候选 | 实际候选与完整 SVD 的弃权 |
|---:|---:|---:|---:|
| 2 | 0.060 ms | 0.109 ms | \(4.61\times10^{-9}\) |
| 4 | 0.302 ms | 0.173 ms | \(2.22\times10^{-8}\) |
| 8 | 1.793 ms | 0.378 ms | \(6.28\times10^{-8}\) |
| 16 | 12.276 ms | 1.069 ms | \(1.38\times10^{-7}\) |

五态 clock quench 的示例从两站点 \(|0\rangle|0\rangle\) 乘积态出发。取 \(Z|k\rangle=e^{2\pi ik/5}|k\rangle\)、\(X|k\rangle=|k+1\bmod 5\rangle\)，局部密度为 \(-Z\otimes Z^\dagger-Z^\dagger\otimes Z-0.8[(X+X^\dagger)\otimes I+I\otimes(X+X^\dagger)]\)。二阶步长 0.05，演化至 \(t=1.2\)。下表用同一步长和 \(\chi_{\max}=32\) 的完整 SVD 结果作参考，误差是末时刻实部 \(\langle Z\rangle\) 的绝对差。时间是三次中位数，包含门应用和截断，排除门准备及测量。每个实验路径有 72 次局部更新。未达标表示该步的目标误差在当前键维上限下无法由所用截断满足，包括回退后仍超标的情况。

| \(\chi_{\max}\) | 路径 | 演化时间 | QR 次数 | 未达标次数 | 末时刻误差 |
|---:|---|---:|---:|---:|---:|
| 8 | 完整 SVD | 17.1 ms | — | — | \(8.32\times10^{-4}\) |
| 8 | QR \(10^{-6}\) | 23.7 ms | 40 | 31 | \(8.66\times10^{-4}\) |
| 8 | QR \(10^{-4}\) | 14.4 ms | 67 | 4 | \(8.54\times10^{-4}\) |
| 12 | 完整 SVD | 49.4 ms | — | — | \(2.32\times10^{-4}\) |
| 12 | QR \(10^{-6}\) | 43.9 ms | 49 | 21 | \(2.32\times10^{-4}\) |
| 12 | QR \(10^{-4}\) | 29.1 ms | 70 | 0 | \(2.32\times10^{-4}\) |

近低秩切口的大 \(d\) 收益明确。端到端收益同时受谱形、容差和回退频率影响。进一步评估默认分派时，需要更长时间和多个观测量的误差对照，以及 TensorKit 内存曲线。

最终回归为 dense 470 项、TensorKit 0.16.5 对称性与 Heisenberg 回归 433 项、TensorKit 0.17.1 对称性基础回归 388 项，均通过。两版使用阶段 A 同一套隔离的 Julia 1.12 定义顺序兼容修正。

## 阶段 D 实施与发布决定

**规范化。** Dense `restore_canonical!(ψ; method=:auto)` 先在候选态上运行 Gram，分别检查右等距条件和左 Schmidt 环境。识别到数值求解失败，或候选规范残差过大时，从原输入运行周期 QR/LQ。Gram 仍是默认方法。周期 QR/LQ 的 mixed-transfer Arnoldi 只提出 seam carry，随后仍用相对三角残差判断闭合。可以用 `accelerate=false` 运行纯扫掠对照。近小谱隙测试比较了同一 64 圈预算下的两种方式，并用原物理态重叠验收。 [规范化入口](../../../src/Schmidt.jl)

**Dense 截断。** 从五态 clock 乘积态演化至 \(t=2.4\)，每 0.4 采样实部 \(\langle Z\rangle\)、能量密度和最大键熵。参考是同模型的完整 SVD，\(dt=0.025,\chi_{\max}=40\)，表里是相对该参考的最大绝对误差，不把有限键维参考视为精确解。比较了 \(dt=0.05,0.025\)、\(\chi_{\max}=12,20\) 和两个候选容差。采样和门准备排除在计时外。 [复现实验](../../../bench/bench_qr_tebd.jl)

| \(dt,\chi_{\max}\) | 后端 | 演化时间 | 接受 QR | 最大误差 \(Z,E,S_{\max}\) |
|---|---|---:|---:|---|
| 0.05, 12 | 完整 SVD | 68.5 ms | — | 0.05410, 0.02493, 0.55552 |
| 0.05, 12 | QR \(10^{-6}\) | 85.5 ms | 49/144 | 0.05410, 0.02493, 0.55552 |
| 0.025, 20 | 完整 SVD | 411.6 ms | — | 0.01333, 0.00578, 0.22398 |
| 0.025, 20 | QR \(10^{-6}\) | 430.6 ms | 143/288 | 0.01305, 0.00606, 0.23917 |
| 0.025, 20 | QR \(10^{-4}\) | 265.4 ms | 252/288 | 0.01306, 0.00605, 0.23934 |

严格容差有回退成本，放宽容差的速度收益也可能伴随熵误差增大。因此 dense QR-TEBD 保留为实验核，`applygate!`、`truncerr` 和 `BondStat` 维持完整 SVD 的截断契约。

**TensorKit。** 实现了按 sector 生成右子空间的实验核。小核仍由 TensorKit 在完整 graded 空间上统一选择被保留的秩，不对每个 sector 各给一份 `maxdim`。候选误差用实际投影张量的范数计算，失败时恢复完整 SVD。一个真实 U(1) 两站点门切口中，两法得到的弃权均为 \(2.04\times10^{-5}\)，完整 SVD 用 0.139 ms，实验 QR 用 0.165 ms。该证据支持保留实验入口。

另用 \(d=5,\chi=16\) 的归一化四站点 graded 晶胞重复构造长晶胞。每个 \(n\) 用独立进程，先在四站点预热。Stage timings 为五次中位数，统计中不含输入复制。峰值 RSS 含 Julia 运行时、JIT 和已预热分配，是整个进程的高水位，不能归因于单个核的瞬时内存。 [复现实验](../../../bench/bench_symmetric_qr.jl)

| \(n\) | Transfer 固定点 | 流式拆分 | 完整规范化累计分配 | 预热后基线 → 测量后峰值 RSS |
|---:|---:|---:|---:|---:|
| 8 | 5.51 ms | 0.57 ms / 1.05 MiB | 9.20 MiB | 1445 → 1463 MiB |
| 16 | 10.99 ms | 1.25 ms / 2.25 MiB | 18.08 MiB | 1434 → 1434 MiB |
| 32 | 33.27 ms | 3.26 ms / 4.66 MiB | 36.12 MiB | 1450 → 1450 MiB |

固定点求解是这些长晶胞的主要耗时。接近线性增长的分配曲线支持保留逐站点分解。在更广泛的模型上仍可对候选 QR 截断做研究，当前结果不足以把它设为默认。

最终 dense 相关回归 658 项通过。TensorKit 0.16.5 和 0.17.1 的基础对称性回归各 394 项通过，两版仍使用阶段 A 的隔离 Julia 1.12 兼容修正，仓库依赖未改。
