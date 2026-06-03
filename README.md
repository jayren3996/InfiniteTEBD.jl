<div align="center">

<img src="docs/src/assets/logo.svg" alt="InfiniteTEBD.jl logo" width="170"/>

# InfiniteTEBD.jl

**Time-evolving block decimation for translationally invariant 1D quantum systems.**

Infinite matrix-product states in the thermodynamic limit, with a built-in ScarFinder search.

[![Docs](https://img.shields.io/badge/docs-latest-9558B2.svg)](https://jayren3996.github.io/InfiniteTEBD.jl/dev/) [![CI](https://github.com/jayren3996/InfiniteTEBD.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/jayren3996/InfiniteTEBD.jl/actions/workflows/CI.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Julia](https://img.shields.io/badge/Julia-1.10%2B-389826.svg)](https://julialang.org)

</div>

---

InfiniteTEBD.jl implements the time-evolving block decimation (TEBD) algorithm for infinite
matrix-product states (iMPS). A state is specified by a periodic unit cell and represented
directly in the thermodynamic limit, so no finite-size extrapolation from a long open chain is
required. This is the right tool when the observables of interest — entanglement structure, bulk
energy density, scar trajectories — are properties of the unit cell rather than of a particular
system size.

The scope is deliberately limited. It implements neither finite-size DMRG nor mixed boundary
conditions, and canonicalization assumes an injective iMPS throughout. For finite-size DMRG,
[`ITensors.jl`](https://github.com/ITensor/ITensors.jl) is the more appropriate choice.

## ✨ Features

|  |  |
| --- | --- |
| 🧩 **Infinite matrix-product states** | Arbitrary periodic unit cells, represented directly in the thermodynamic limit. |
| 🌀 **Local-gate evolution** | `applygate!` / `evolve!` with a discarded-weight truncation controller. |
| ⏱️ **Trotter helpers** | Second-order Strang splitting and two fourth-order schemes (`trotter_gates`). |
| 📐 **Schmidt canonicalization** | Injective-setting canonical form via `canonical!`. |
| 📊 **Transfer-matrix observables** | Overlaps, multi-site expectation values, entanglement entropy, and energy density. |
| 🎯 **ScarFinder search** | Low-entanglement trajectories via Hamiltonian, gate, and mixed interfaces (`scarfinder!`). |
| ⚛️ **Abelian-symmetric tensors** | `:U1`, `:Z2`, and products through an optional TensorKit extension. |
| 🪜 **Adaptive bond dimension** | A ratchet on bond growth (`adaptive_bonddim`, `natural_bonddim`). |

## 📦 Installation

`InfiniteTEBD` is registered in Julia's General registry, so install it by name:

```julia
pkg> add InfiniteTEBD
```

Then load it with `using InfiniteTEBD`. Requires Julia 1.10+; `TensorKit.jl` 0.16 is needed only
for the symmetric extension.

> **Note on the name.** This library was previously distributed, unregistered, as `iTEBD`. Julia's
> General registry requires a mixed-case name, so the registered package is `InfiniteTEBD`. The
> original [`iTEBD.jl`](https://github.com/jayren3996/iTEBD.jl) repository is kept as a thin
> compatibility shim that re-exports `InfiniteTEBD` — existing code that installs it by URL and
> calls `using iTEBD` keeps working unchanged. New code should use `InfiniteTEBD`.

## 🚀 Quick Start

Imaginary-time iTEBD relaxes a random spin-1 state into the AKLT ground state in about a hundred
Trotter steps:

```julia
using InfiniteTEBD, LinearAlgebra

# Spin-1 operators
X = sqrt(2)/2 * [0 1 0; 1 0 1; 0 1 0]
Y = sqrt(2)/2 * 1im * [0 -1 0; 1 0 -1; 0 1 0]
Z = [1 0 0; 0 0 0; 0 0 -1]

# AKLT bilinear-biquadratic Hamiltonian density
SS = kron(X, X) + kron(Y, Y) + kron(Z, Z)
H  = 0.5 * SS + SS^2 / 6 + I / 3

# Two-site unit cell, imaginary-time evolution
psi   = rand_iMPS(ComplexF64, 2, 3, 1)
gates = [(exp(-0.1 * H), 1, 2), (exp(-0.1 * H), 2, 1)]
evolve!(psi, gates, 300; maxdim=8)

energy_density(psi, H)         # → ≈ 0.0  (AKLT ground state)
maximum(length.(psi.λ))        # → 2      (converged bond dimension)
```

A walkthrough of this example with more diagnostics is on the
[Time Evolution](https://jayren3996.github.io/InfiniteTEBD.jl/dev/time-evolution/) page.

## 🧭 Choosing Your Path

| If you want to … | Start with |
|------------------|------------|
| Build your first infinite MPS | [Getting Started](https://jayren3996.github.io/InfiniteTEBD.jl/dev/getting-started/) → [States and Canonical Form](https://jayren3996.github.io/InfiniteTEBD.jl/dev/imps/) |
| Evolve a state in real or imaginary time | [Time Evolution](https://jayren3996.github.io/InfiniteTEBD.jl/dev/time-evolution/) |
| Measure energy, entropy, or overlaps | [Observables](https://jayren3996.github.io/InfiniteTEBD.jl/dev/observables/) |
| Exploit a U(1) or Zₙ symmetry | [Symmetric infinite MPS](https://jayren3996.github.io/InfiniteTEBD.jl/dev/symmetries/) |
| Search for low-entanglement scar trajectories | [ScarFinder Workflow](https://jayren3996.github.io/InfiniteTEBD.jl/dev/scarfinder/) |
| Look up exact signatures and defaults | [API Reference](https://jayren3996.github.io/InfiniteTEBD.jl/dev/api/) |

## 🛠 Usage

<details>
<summary><b>The iMPS storage convention</b></summary>

<br>

The single rule to remember: **the stored tensor has the right Schmidt values already multiplied
in.** Formally, after `canonical!`,

```
B_i = Γ_i · λ_i
```

so `psi.Γ[i]` returns the right-canonical tensor `B_i`, not the bare Vidal `Γ_i`. To get the Vidal
pair, index the state: `Γ_i, λ_i = psi[i]`. The
[States and Canonical Form](https://jayren3996.github.io/InfiniteTEBD.jl/dev/imps/) page explains
this in detail.

</details>

<details>
<summary><b>ScarFinder: low-entanglement trajectories</b></summary>

<br>

The package includes a general ScarFinder workflow that searches for low-entanglement,
weakly-thermalizing trajectories directly on the iMPS manifold. Each iteration evolves the state
for a short real-time interval, projects back to a target bond dimension `χ`, and optionally
applies a small imaginary-time correction to hold the energy density near a chosen target.

Three interfaces are exposed:

```julia
scarfinder!(ψ, h, dt, χ, N; ...)    # Hamiltonian-based
scarfinder!(ψ, G, χ, N; ...)        # gate-based, no energy correction
scarfinder!(ψ, G, h, χ, N; ...)     # mixed: custom gate G, energy fixed against h
```

The mixed form is the recommended one for constrained models like PXP, where the gate carries
projectors that repair truncation artifacts while the energy target is defined against the
unprojected Hamiltonian density. See the
[ScarFinder Workflow](https://jayren3996.github.io/InfiniteTEBD.jl/dev/scarfinder/) page for the
complete PXP example.

</details>

<details>
<summary><b>Abelian-symmetric tensors via TensorKit</b></summary>

<br>

Abelian-symmetric tensors (`:U1`, `:Z2`, and products) are supported through a TensorKit-based
extension, loaded automatically once you bring TensorKit into scope:

```julia
using InfiniteTEBD, TensorKit
```

The symmetric API mirrors the dense one. See
[Symmetric infinite MPS](https://jayren3996.github.io/InfiniteTEBD.jl/dev/symmetries/) for the
charge-sector conventions and worked examples.

</details>

## 📚 Documentation

Full documentation lives at
**[jayren3996.github.io/InfiniteTEBD.jl](https://jayren3996.github.io/InfiniteTEBD.jl/dev/)**.

- [Getting Started](https://jayren3996.github.io/InfiniteTEBD.jl/dev/getting-started/) — installation, the first state, sanity checks.
- [States and Canonical Form](https://jayren3996.github.io/InfiniteTEBD.jl/dev/imps/) — the storage convention, `canonical!`, runnable examples.
- [Symmetric infinite MPS](https://jayren3996.github.io/InfiniteTEBD.jl/dev/symmetries/) — Abelian U(1)/Zₙ tensors via the TensorKit extension.
- [Time Evolution](https://jayren3996.github.io/InfiniteTEBD.jl/dev/time-evolution/) — `applygate!`, `evolve!`, Trotter order, adaptive bond dimension.
- [Observables](https://jayren3996.github.io/InfiniteTEBD.jl/dev/observables/) — overlaps, expectation values, entropy, energy density.
- [ScarFinder Workflow](https://jayren3996.github.io/InfiniteTEBD.jl/dev/scarfinder/) — the three interfaces, the two time scales, the PXP example.
- [API Reference](https://jayren3996.github.io/InfiniteTEBD.jl/dev/api/) — generated from docstrings.

## 🗂 Examples

Runnable Jupyter notebooks live in [`examples/`](examples):

- [`CanonicalForm.ipynb`](examples/CanonicalForm.ipynb) — the storage convention and `canonical!` in practice.
- [`AKLT_GS.ipynb`](examples/AKLT_GS.ipynb) — imaginary-time relaxation into the AKLT ground state.
- [`PXP.ipynb`](examples/PXP.ipynb) — real-time evolution in the constrained PXP model.
- [`PXP_ScarFinder.ipynb`](examples/PXP_ScarFinder.ipynb) — the full ScarFinder workflow on PXP.

## 🧱 Repository Map

| Path | What lives there |
|------|------------------|
| [`src/InfiniteTEBD.jl`](src/InfiniteTEBD.jl) | Top-level module; includes and re-exports the public API |
| [`src/iMPS.jl`](src/iMPS.jl) | The `iMPS` type, constructors, `canonical!`, `expect`, `ent_S` |
| [`src/Gate.jl`](src/Gate.jl) | Local-gate updates and Trotter helpers (`applygate!`, `evolve!`, `trotter_gates`) |
| [`src/Schmidt.jl`](src/Schmidt.jl) | Schmidt decomposition and the discarded-weight truncation controller |
| [`src/Contractions.jl`](src/Contractions.jl), [`src/TensorAlgebra.jl`](src/TensorAlgebra.jl) | Transfer-matrix factors and the low-level tensor contractions |
| [`src/Krylov.jl`](src/Krylov.jl) | Dominant transfer-matrix eigenvalue via KrylovKit |
| [`src/ScarFinder.jl`](src/ScarFinder.jl) | `scarfinder!`, `energy_density`, `energy_span`, `operator_span` |
| [`src/Miscellaneous.jl`](src/Miscellaneous.jl) | `inner_product` and the adaptive bond-dimension helpers |
| [`src/SymmetricStubs.jl`](src/SymmetricStubs.jl) | Entry points for the symmetric API that require TensorKit |
| [`ext/InfiniteTEBDTensorKitExt.jl`](ext/InfiniteTEBDTensorKitExt.jl) | TensorKit-backed Abelian-symmetric implementation |
| [`docs/`](docs) | Documenter manual and worked examples |
| [`examples/`](examples) | Runnable Jupyter notebooks |

## 📝 Citation

If the ScarFinder routines are useful in your published work, please cite:

```bibtex
@article{Ren2025ScarFinder,
  title   = {ScarFinder: A Detector of Optimal Scar Trajectories in
             Quantum Many-Body Dynamics},
  author  = {Ren, Jie and Hallam, Andrew and Ying, Lei and Papi\'c, Zlatko},
  journal = {PRX Quantum},
  volume  = {6},
  pages   = {040332},
  year    = {2025},
  doi     = {10.1103/PRXQuantum.6.040332},
}
```

## 🙏 Acknowledgements

InfiniteTEBD.jl builds on [`ITensors.jl`](https://github.com/ITensor/ITensors.jl) /
[`ITensorMPS.jl`](https://github.com/ITensor/ITensorMPS.jl) for tensor-network primitives,
[`TensorOperations.jl`](https://github.com/Jutho/TensorOperations.jl) for contractions, and
[`KrylovKit.jl`](https://github.com/Jutho/KrylovKit.jl) for the dominant transfer-matrix
eigenvalue.

## 📄 License

[MIT](LICENSE) © 2026 Jie Ren and contributors.
