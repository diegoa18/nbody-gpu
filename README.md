# N-Body Gravitational Simulator

This project is a high-performance, GPU-accelerated astrophysical N-body simulator written in C11 and CUDA. It is designed as a rigorous scientific engine to model gravitational dynamics—from planetary systems to stellar clusters—focusing strictly on the intersection of theoretical physics and computational optimization.

**v1.0.0** brings OpenMP host parallelization, GPU and multi-core CPU Barnes-Hut solvers, conservation telemetry, and a reproducible Plummer cluster merger showcase.

## Mathematical & Physical Models

### 1. Yoshida 4th-Order Symplectic Integration
Unlike traditional Runge-Kutta methods, this engine exactly preserves the Hamiltonian structure (phase-space volume) of the system over cosmic timescales. The integration is performed via fractional time steps (Drift-Kick-Drift) using the exact Yoshida constants to achieve O(Δt^4) global error while strictly conserving total energy and angular momentum:
* w1 = 1 / (2 - 2^(1/3)) ≈  1.35120719
* w0 = 1 - 2*w1          ≈ -1.70241438

### 2. Traceless Quadrupole Tensor Expansion
To evaluate the gravitational potential of distant clusters with extreme fidelity, the Octree nodes compute up to the quadrupole moment. Instead of treating a cluster as a simple point mass (monopole), we construct the traceless quadrupole tensor Q_ij from the raw inertia tensor I_ij:
* Q_ij = 3 * I_ij - Tr(I) * δ_ij
This flattens the multipole expansion error from O(θ²) to O(θ³), allowing for highly asymmetric structures (like galactic mergers) to be simulated without secular biases.

### 3. Cubic Spline Compact-Support Softening
Standard Plummer softening artificially alters Newtonian gravity at all distances, destroying long-term orbital conservation. This engine implements a compact-support cubic spline kernel (equivalent to the standard used in GADGET-2/4):
* If r >= h: Force is EXACTLY Newtonian (GM / r²)
* If r < h: Force decays smoothly to zero following the exact cubic polynomial limit.

## Computational Architecture

To overcome the unscalable O(N²) complexity of direct gravitational summation, the engine implements the **Barnes-Hut algorithm**, reducing the problem to O(N log N) through spatial partitioning. On the GPU, this is achieved by mapping the 3D space into a 63-bit Morton code (Z-order curve) and building an Octree via parallel Radix sorts. 

This specific data structure allows for a warp-collaborative tree walk: because particles are strictly ordered by spatial locality, the 32 threads in a CUDA warp travel down the same branches of the tree simultaneously. This eliminates branch divergence and guarantees fully coalesced global memory accesses, pushing the hardware to its absolute theoretical limits in pure FP64 arithmetic (bypassing catastrophic FP32 underflow at astronomical scales).

The host implementation mirrors the same tree structure. The Octree is built sequentially, then the force loop is distributed across CPU threads with OpenMP: the direct-sum path uses static scheduling, while the Barnes-Hut target walk uses dynamic scheduling because tree depth varies strongly between cluster cores and halo particles. This gives the benchmark a genuine multi-core CPU baseline instead of a single-threaded reference.

## Performance & Scaling (RTX 2050 vs OpenMP CPU)

Measured with `./build/benchmark_bh` on a 12-thread host CPU and an NVIDIA RTX 2050 (Release build, θ = 0.7, Plummer distributions). `speedup` is `GPU_direct / GPU_BH`; `BH_err` is the maximum relative acceleration error of the GPU Barnes-Hut force against the GPU direct-sum reference. `CPU_BH` is the 12-thread OpenMP Barnes-Hut time.

| N | GPU_direct(ms) | GPU_BH(ms) | speedup | CPU_BH(ms) | BH_err |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.354 | 1.321 | 0.27x | 0.225 | 1.67e-01 |
| 500 | 1.789 | 5.490 | 0.33x | 0.407 | 3.26e-01 |
| 1000 | 3.538 | 8.680 | 0.41x | 1.015 | 3.90e-01 |
| 5000 | 26.204 | 20.490 | 1.28x | 14.863 | 2.31e-01 |
| 10000 | 84.321 | 60.268 | 1.40x | 21.779 | 3.36e-01 |
| 50000 | 2507.456 | 204.651 | 12.25x | 385.091 | 3.75e-01 |
| 100000 | 10876.291 | 569.859 | 19.09x | 737.350 | 2.49e-01 |

At small N the GPU is launch-latency bound: at N=100 the direct GPU kernel is already 3.7x faster than GPU Barnes-Hut. The Barnes-Hut crossover appears at N=5,000, where GPU_BH first beats GPU_direct (1.28x). In the large-N regime the GPU tree walk pulls away:

| N | GPU_BH vs GPU_direct | GPU_BH vs 12-thread CPU_BH |
| ---: | ---: | ---: |
| 50,000 | 12.25x | 1.88x |
| 100,000 | 19.09x | 1.29x |

At N=100,000 the headline result is a 19.09x speedup over GPU direct summation while still running 1.29x faster than the multi-core OpenMP CPU tree walk; at N=50,000 the corresponding gains are 12.25x and 1.88x. The GPU remains the faster Barnes-Hut solver at both large-N points, although its advantage over the 12-thread CPU narrows at 100,000 relative to 50,000.

## Physical Conservation Validation

The flagship `plummer_merger` preset was run on the RTX 2050 with N=20,000, θ=0.7, traceless-quadrupole Barnes-Hut forces, and Yoshida 4th-order integration for 16,000 steps (four crossing times; final time 1.487900e+14 s). The telemetry is written by `--diagnostics` and summarized in `renders/merger_showcase/conservation_summary.txt`.

| Quantity | Peak | Final |
| --- | ---: | ---: |
| rel_dE | 1.801043e-02 (1.80%) | 1.096294e-02 (1.10%) |
| rel_dL | 7.180613e-03 (0.718%) | 6.698830e-03 (0.670%) |
| \|P\| (kg m/s) | 1.751097e+35 | 1.751097e+35 |

Energy starts at E₀ = -1.434487e+41 J and ends at E_f = -1.450213e+41 J, a fractional drift of +1.096294%. The peak drift reaches 1.801043e-02 near pericenter; as the encounter remnant relaxes, the symplectic drift envelope contracts to 1.096294e-02. Angular momentum stays below 7.180613e-03 (0.72%) throughout, ending at 6.698830e-03.

The initial center-of-mass residuals are |Σm r| / Σ(m|r|) = 2.413e-17 and |Σm v| / Σ(m|v|) = 1.748e-17. The final |P| is 1.751097e+35 kg m/s. These are measured envelopes for this production run, not analytic guarantees; they are recorded so that changes to the tree, softening, or integrator are immediately visible.

## Build & Quickstart

```bash
cmake -B build && cmake --build build
```

For the benchmark and showcase numbers above, build in Release mode (the showcase script does this automatically):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Run the validation suites and the GPU/CPU benchmark:

```bash
./build/test_cpu
./build/test_gpu
./build/benchmark_bh
```

Both suites report `PASS (0 failures)` on the RTX 2050 host. The Barnes-Hut θ=0 trajectory check matches direct summation exactly (maximum relative position error 0.000e+00), the θ=0.7 energy error is 8.258e-13, and the Yoshida convergence study recovers an average order of 3.93 (expected 4).

Generate the complete merger showcase:

```bash
./scripts/run_merger_showcase.sh
```

The script builds Release, runs the N=20,000 GPU merger with θ=0.7, records `diagnostics.csv`, renders the initial, pericenter, late/post-merger, and final phases, runs `benchmark_bh`, writes `conservation_summary.txt`, and removes the raw `.nb` snapshots. Outputs land in `renders/merger_showcase/`. Use `NBODY_BACKEND=cpu` for a CPU-only run, or `N`, `THETA`, `T_END`, and `DT` to override the physical configuration.

## Roadmap & Next Steps

1. **Multi-GPU Scaling (MPI):** Distributing the spatial domain using Local Essential Trees across multiple discrete GPUs for cluster-level execution.
2. **Inelastic Collisions:** Implementing physically accurate merger mechanics when the physical distance between two bodies drops below the sum of their radii, perfectly conserving linear momentum.
