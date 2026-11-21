# iTEBD 的大晶胞、长程门与大物理维数

2026-11-21

优先把逐站点规范化接入真正会触发整块合并的路径，然后为远距离两端点门提供专门的更新。两项改动解决不同的指数增长。大物理维数下单个相邻门和截断的代价仍然会增长，要根据算符结构分别处理。

## 晶胞长度和门支撑是两个尺度

设晶胞长度为 `n`、单点维数为 `d`、典型键维为 `χ`，门的连续支撑长度为 `ℓ`。作用距离、Trotter 分层的周期、态本身的平移周期也需要分别处理。长程平移不变哈密顿量可以有较短的物理周期，但显式周期 SWAP 路由需要防止不同站点绕回成同一张存储张量。

| 路径 | 主要成本及来源 |
|---|---|
| 整晶胞规范化 | `tensor_group` 生成 `χ × d^n × χ`。完整 SVD 的第一次拆分约为 `(χd) × (χd^(n−1))`。发生截断时 `canonical!` 可能再做一轮。 |
| 逐站点规范化 | 一次链式 transfer matvec 为 `O(ndχ³)`，再乘左右固定点求解各自需要的 matvec 数。存储为本地张量、逐键环境和 Krylov 基，随 `n` 线性增长。 |
| 连续稠密门 | 门矩阵占 `O(d^(2ℓ))`，融合的态块占 `O(χ²d^ℓ)`。一般稠密门收缩约为 `O(χ²d^(2ℓ))`，通用稠密矩阵指数约为 `O(d^(3ℓ))`。之后还有逐键 SVD。 |
| 相邻稠密两站点门 | 门应用约为 `O(χ²d⁴)`，完整 SVD 约为 `O((χd)³)`。当前先做完整 SVD，再按 `maxdim` 或 discarded-weight 目标取子空间。 |

`applygate!(ψ,G,i,j)` 使用整个周期区间，包含两个端点之间的站点。即便物理上只有端点相互作用，补单位算符后也会付连续稠密门的代价。例子中 `d=8, ℓ=6` 的 `ComplexF64` 稠密门单独占 1 TiB，两端点矩阵只占 64 KiB。 [门收缩与拆分](../../src/Gate.jl#L55) · [周期支撑](../../src/Gate.jl#L259) · [完整 SVD](../../src/TensorAlgebra.jl#L558)

## 本地测量

Julia 1.12.6，单线程 BLAS，种子 `20261121`。每点预热后取五次中位数。输入为随机 `ComplexF64` dense iMPS。同一参数对照用相同输入。使用 `@timed`，分配量是整次调用的累计分配，峰值内存需要另测。计时排除了输入复制、编译和门构建。

### 纯规范化

固定 `χ=12`。整块路径使用 `canonical!(; maxdim=12, cutoff=1e-12, noninjective=:ignore)`，逐站点路径使用 `restore_canonical!`。整块张量的最低存储估计超过 64 MiB 时跳过整块实测。

| 晶胞 `n` | `d` | 单个融合张量 | 整块时间 / 累计分配 | 逐站点时间 / 累计分配 |
|---:|---:|---:|---:|---:|
| 4 | 3 | 0.178 MiB | 8.65 ms / 26.0 MiB | 2.21 ms / 4.0 MiB |
| 6 | 4 | 9 MiB | 510 ms / 1,187 MiB | 3.97 ms / 7.4 MiB |
| 7 | 4 | 36 MiB | 1,911 ms / 4,742 MiB | 3.62 ms / 8.6 MiB |
| 8 | 4 | 144 MiB | 跳过 | 3.87 ms / 9.7 MiB |
| 12 | 4 | 36 GiB | 跳过 | 7.14 ms / 14.5 MiB |

三个直接对照的每晶胞归一化重叠在所报精度下均为 1.0000000000。逐站点方法的最大右等距残差约为 `10^-14`。这是温和条件数的随机态测试，实际长时间演化的固定点迭代数还取决于 transfer 谱隙和环境条件数。 [现有逐站点实现](../../src/Schmidt.jl#L343)

### 单个远端点门

六站点晶胞，`d=4`、初始 `χ=8`，每次更新的上限为 12，`dt=0.1`。同一个随机 Hermitian 两端点生成元构造门。稠密路径补齐中间单位算符后一次更新整个连续区间。SWAP 路径用现有的相邻稠密 `applygate!` 将远端搬到近端，施加门，再按逆序搬回。重叠以每晶胞的混合 transfer 主特征值模除以两态自 transfer 主特征值乘积的平方根。

| 连续支撑 `ℓ` | 稠密块 ms / MiB | SWAP ms / MiB | 两结果归一化重叠 |
|---:|---:|---:|---:|
| 2 | 0.192 / 0.225 | 0.140 / 0.225 | 1.000000 |
| 3 | 0.568 / 0.836 | 0.566 / 0.776 | 0.947778 |
| 4 | 3.211 / 3.727 | 1.241 / 1.567 | 0.863032 |
| 5 | 22.523 / 25.931 | 1.512 / 2.358 | 0.799724 |

另一个 `d=2, ℓ=4, χ_in=2, maxdim=64` 的小例子，两条路径的重叠为 0.9999999999999996。相同键维下的差异来自不同的截断位置与顺序。有限键维表格比较的是两个输出，未判定哪个更接近精确态。这些数据只覆盖单次更新，完整周期调度和长时间误差需要单独验证。

## 规范化的改动顺序

1. **先封住隐式分组。** `product_iMPS` 已把每个局部向量归一化并设 `λ=1`，可验证直接返回所构造的态。一般构造器、`canonical!`、跨缝 `recanonicalize=true` 和 `chi_policy=:adaptive` 都仍可能进整块规范化，自适应路径甚至可在每个多站点门后触发。现有压缩分组保护只看 `n` 和 `d^n`，`d=8,n=6,χ=64` 会通过，单个融合张量已需 16 GiB。应按元素类型和两端键维预估字节数，为临时副本留余量。TensorKit 的固定点已经逐站点求，后续 split 仍可能合并整胞。 [构造器](../../src/iMPS.jl#L261) · [自适应更新](../../src/Gate.jl#L588) · [现有保护](../../src/ScarFinder.jl#L53) · [TensorKit split](../../ext/InfiniteTEBDTensorKitExt.jl#L929)
2. **让纯 gauge 恢复走现有 `restore_canonical!`。** 它固定键维并归一化。真正需要降低 `χ` 时，逐键 SVD 后再做逐站点 gauge 恢复是一条内存有界的候选路径。两种压缩的有限 `χ` 输出可能不同，需测 discarded weight、观测量及收敛，并保留对原有秩处理、norm 与非 injective 语义的检查。还可以利用已有 `ChainTransferWorkspace`，补齐左向缓存，并在键空间兼容时 warm-start 环境；避免固定点失败时退回不可承受的稠密构造。 [压缩和 gauge 的现有组合](../../src/ScarFinder.jl#L167) · [Transfer workspace](../../src/TensorAlgebra.jl#L969) · [稠密回退](../../src/Miscellaneous.jl#L259)
3. **近奇异环境参考 QR/polar 路径。** TeNPy 在一个 seam 求左右固定点，再局部扫过晶胞，也提供由 QR 迭代求规范形的方法。ITensorInfiniteMPS 用 transfer 固定点与逐站点 polar sweep。MPSKit 在局部正交扫掠后加入固定点加速。左右 Gram 对小 Schmidt 值更敏感，单层 QR/polar 适合作为病态案例的替代方案。简单地扫一轮并丢弃 wrap 因子会改变物理态，测试应检查规范条件和态重叠。 [Orús–Vidal](https://arxiv.org/abs/0711.3960) · [Vanderstraeten 等，`2.3](https://arxiv.org/pdf/1810.07006) · [TeNPy](https://github.com/tenpy/tenpy/blob/main/tenpy/networks/mps.py#L4281-L4427) · [ITensorInfiniteMPS](https://github.com/ITensor/ITensorInfiniteMPS.jl/blob/main/src/orthogonalize.jl#L4-L122) · [MPSKit](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/src/states/ortho.jl#L201-L328)

## 远端点门与大 `d`

- **给两端点门单独的底层入口。** 距离 `r` 的 SWAP 路由需要 `2(r−1)+1` 次相邻更新，在每步键维有界时对距离是线性的。SWAP 本身可以用物理腿置换实现，门序也可以消去相邻的逆置换。路由过程会改变中间切口的纠缠，宜作为验证基线或可选后端。实现周期版时，分别测试跨缝、反向路径、重复键、并行层的重叠和 fermionic SWAP 的符号。ITensorMPS 和 quimb 的有限链路由可以借鉴，其实现不能直接确定无限周期更新的调度。 [Stoudenmire–White，`III.A.2](https://arxiv.org/pdf/1002.1305) · [ITensorMPS](https://github.com/ITensor/ITensorMPS.jl/blob/main/src/abstractmps.jl#L1874-L1917) · [quimb](https://quimb.readthedocs.io/en/latest/circuit/circuit-mps.html)
- **低秩门优先考虑精确短 MPO。** 将端点门写成 `G = Σα Aα ⊗ Bα`，中间传递 MPO 指标并作用物理单位算符。单门秩 `w≤d²` 与距离无关。对角两端点相位门只需分解 `d×d` 相位表，`w≤d`。若 `P²=I`，`exp(−iθ P⊗P)` 的秩至多为 2。这个表示在压缩前与原门相同，朴素应用会把跨过的键扩到约 `wχ`，多条门路径相交时秩还可能相乘。先测单门并在相同误差门槛下比较 SWAP 和低秩 MPO，再决定局部 sweep 还是无限变分压缩。远距离观测量也应把中间区间改成 transfer sweep。 [quimb MPO 门接口](https://quimb.readthedocs.io/en/main/autoapi/quimb/tensor/tn1d/core/index.html#quimb.tensor.tn1d.core.MatrixProductState.gate_nonlocal) · [Vanhecke 等](https://arxiv.org/abs/2001.11882) · [本库观测量收缩](../../src/Contractions.jl#L52)
- **大 `d` 把门应用和截断分开优化。** 对角门应直接作用在物理腿上，近邻相位表的应用约 `O(χ²d²)`。剩余热点若是 SVD，可试 Unfried 等人的 QR-TEBD，截断的 `d` 标度由三次降为二次。TeNPy 有相应实现，其近邻更新还演示如何避免显式除以很小的左 Schmidt 值。想保留本库的 discarded-weight 接口，部分 SVD 需要检查尾权重、谱聚簇与残差。玻色子或振动子模型若单点约化谱可压缩，再考虑动态局部基，单独记录物理基丢弃权重。 [QR-TEBD](https://arxiv.org/html/2212.09782) · [TeNPy QR 实现](https://github.com/tenpy/tenpy/blob/main/tenpy/algorithms/tebd.py#L574-L682) · [TeNPy 相邻更新](https://github.com/tenpy/tenpy/blob/main/tenpy/algorithms/tebd.py#L409-L439) · [TEBD-LBO](https://arxiv.org/abs/1508.00694)
- **大量长程项适合另测整步演化。** Hamiltonian 有紧凑 MPO 时，可试 Zaletel 等的 `W^I/W^II` 时间 MPO，或者按胞内、胞间分块，以 MPS–Krylov 做局部时间步的 h-iTEBD。若愿意直接演化固定键维流形，可另测 uniform TDVP。它们的时间步或变分误差应独立测量，尤其是 ScarFinder 有意保留的演化与投影间隔。 [Zaletel 等](https://arxiv.org/abs/1407.1832) · [h-iTEBD](https://arxiv.org/pdf/1910.10726) · [TDVP](https://arxiv.org/abs/1408.5056) · [MPSKit 无限 TDVP](https://github.com/QuantumKitHub/MPSKit.jl/blob/main/src/algorithms/timestep/tdvp.jl#L93-L138)

下一轮实验可用三个对照。首先在大晶胞测试归一化重叠、左右规范残差、逐键 Schmidt 谱和峰值内存。其次用低秩对角门及一般两端点门，在有限小环的精确参考上校验方向和周期语义，再按观测误差匹配键维。最后固定物理时间和 ScarFinder 的投影间隔，分别扫描 `dt`、键维、路由截断及 MPO 压缩容差，从而分辨 Trotter 误差与各类截断误差。

## Dense 实施与复测

- **规范化 dispatch。** `product_iMPS` 直接返回规范化好的乘积态。一般 dense `canonical!` 在单个融合张量预计超过 1 MiB 时切换到逐站点实现，构造器和现有演化中的规范化调用一起受益。这个阈值用于选择算法，整块基准中累计分配量远大于单个融合张量的大小。逐站点固定点选择较小的虚键，纯 gauge 恢复保持键维。确需截断时，自动路径在恢复前后两次 gauge 之间，对各键同时保留目标 Schmidt 子空间。这个截断定义与旧的整块 SVD 扫掠不同。小 dense 晶胞的 sector selection 仍走原实现。大晶胞 dense 路径限于有唯一物理 transfer 固定点的态。 [规范化文档](../src/imps.md)
- **分离端点门和连续门。** `EndpointGate` 可以预先分解门，直接放入现有 `evolve!` 的门列表，或用于 `expect`。默认后端把端点因子通过中间站点传递，先做无截断的右向等距化，再按 dense 核的切口顺序做截断 SVD。SWAP 是显式选择，物理腿置换无需构造稠密 SWAP 矩阵。`Diagonal` 门在端点分解时使用小相位表，连续更新时也只置换和乘对角元素。 [门使用说明](../src/time-evolution.md)

复测使用六站点 `d=4` dense 态，输入 `χ=8`、上限 12。Julia 1.12.6，单线程 BLAS，预热后七次取中位数。门和 MPO 因子提前生成，计时排除输入复制。MPO 结果对同一输入的稠密块结果，在所报精度下每晶胞归一化重叠均为 1。

| 门及包围区间 | 稠密块 ms / 累计 MiB | 预生成 MPO ms / 累计 MiB | SWAP ms / 累计 MiB |
|---|---:|---:|---:|
| 秩 2 相位门，4 站点 | 2.715 / 3.727 | 0.195 / 0.516 | 0.851 / 1.275 |
| 秩 2 相位门，5 站点 | 20.866 / 25.931 | 0.268 / 0.701 | 1.222 / 1.915 |
| 秩 16 一般门，4 站点 | 2.613 / 3.727 | 6.538 / 8.269 | 0.808 / 1.327 |
| 秩 16 一般门，5 站点 | 21.438 / 25.931 | 11.050 / 13.501 | 1.253 / 1.966 |

低秩下无需付出长区间融合代价。满算符秩、短距离时，MPO 扫掠的中间键维也可能让它比直接块更新慢。两条后端的选择应由实际支撑跨度、算符秩和截断误差决定。当前实现范围是 dense 端点门和逐站点规范化。TensorKit 的整胞拆分、病态环境的 QR/polar 备选路径以及大 `d` 通用门的 QR-TEBD 另需各自算法和验收。
