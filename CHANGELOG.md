# Changelog

## 0.3.0

Layer MPOs and ScarFinder building blocks. The existing API is unchanged.

### Added

- `iMPO`: infinite MPO with tensors `(Dl, d_out, d_in, Dr)` and cell length
  `n`; `bonddims`, operator product `A * B` / `mpo_multiply` with optional
  compression, `compress_mpo`.
- `layer_mpo(layer, n)`: exact infinite MPO of a layer of commuting local gates
  by operator Schmidt decomposition. Supports may be longer than the cell
  (`(i, j)` with `i ≤ j` are chain sites). `check_commuting(layer; n)` verifies
  the precondition; violations throw.
- `apply!(ψ, W)`: MPO application followed by canonical form and per-bond
  truncation (`maxdim`, `cutoff`, `truncerr`), with `evolve!`-style stats;
  works for non-unitary MPOs.
- `trotter_mpos(layers, dt; trotter, evolution, n, gate_map)` and
  `evolve!(ψ, mpos::Vector{<:iMPO}, steps)`, which merges the half steps at
  step boundaries.
- `truncate!(ψ, χ)`: bond-symmetric truncation (extends `ITensorMPS.truncate!`).
- `energy_density(ψ, terms; n)` for site-dependent terms, `energy_fix!` with
  imaginary-time layer MPOs returning `(converged, residual, steps)`,
  `per_site_fidelity`, and `scarfinder_step!(ψ, mpos, χ; …)` returning
  `(loss, energy_error, converged)`.

### Notes

- `apply!` follows Orús–Vidal (contract, canonicalize, truncate all bonds) with
  the single-layer QR canonicalization of Vanderstraeten–Haegeman–Verstraete
  (Algorithms 1–2) after an exact removal of null directions. Schmidt values
  are resolved to ~1e-15 relative; the fixed-point `canonical!` resolves ~1e-8.
- `truncate_unitcell!` is unchanged; its docstring now contrasts it with
  `truncate!`.
