# N-Body Gravitational Simulator

This project is a high-performance, GPU-accelerated astrophysical N-body simulator written in C11 and CUDA. It is designed as a rigorous scientific engine to model gravitational dynamics—from planetary systems to stellar clusters—focusing strictly on the intersection of theoretical physics and computational optimization.

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

## Roadmap & Next Steps

1. **CPU Parallelization:** Injecting OpenMP pragmas into the host-side tree-walk and integration loops to saturate modern multi-core processors.
2. **Multi-GPU Scaling (MPI):** Distributing the spatial domain using Local Essential Trees across multiple discrete GPUs for cluster-level execution.
3. **Inelastic Collisions:** Implementing physically accurate merger mechanics when the physical distance between two bodies drops below the sum of their radii, perfectly conserving linear momentum.
