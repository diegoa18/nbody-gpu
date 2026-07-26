#include "gpu_context.h"
#include "nbody/forces.h"
#include "nbody/constants.h"

//euler explicito: x+=v*dt, v+=a*dt
__global__ void integr_euler_kernel(double * __restrict__ px,
                                    double * __restrict__ py,
                                    double * __restrict__ pz,
                                    double * __restrict__ vx,
                                    double * __restrict__ vy,
                                    double * __restrict__ vz,
                                    double * __restrict__ ax,
                                    double * __restrict__ ay,
                                    double * __restrict__ az,
                                    index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;
}

// euler semi-implicito: v+=a*dt, x+=v*dt (simplectico)
__global__ void integr_semiimplicit_kernel(double * __restrict__ px,
                                           double * __restrict__ py,
                                           double * __restrict__ pz,
                                           double * __restrict__ vx,
                                           double * __restrict__ vy,
                                           double * __restrict__ vz,
                                           double * __restrict__ ax,
                                           double * __restrict__ ay,
                                           double * __restrict__ az,
                                           index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}

//velocity verlet
__global__ void integr_verlet_pos_kernel(double * __restrict__ px,
                                         double * __restrict__ py,
                                         double * __restrict__ pz,
                                         double * __restrict__ vx,
                                         double * __restrict__ vy,
                                         double * __restrict__ vz,
                                         double * __restrict__ ax,
                                         double * __restrict__ ay,
                                         double * __restrict__ az,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    /* x(t+dt) = x(t) + v(t)·dt + 0.5·a(t)·dt² */
    px[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
    py[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
    pz[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
}

__global__ void integr_verlet_vel_kernel(double * __restrict__ vx,
                                         double * __restrict__ vy,
                                         double * __restrict__ vz,
                                         double * __restrict__ ax_old,
                                         double * __restrict__ ay_old,
                                         double * __restrict__ az_old,
                                         double * __restrict__ ax_new,
                                         double * __restrict__ ay_new,
                                         double * __restrict__ az_new,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    /* v(t+dt) = v(t) + 0.5*(a_old + a_new)·dt */
    vx[i] += 0.5 * (ax_old[i] + ax_new[i]) * dt;
    vy[i] += 0.5 * (ay_old[i] + ay_new[i]) * dt;
    vz[i] += 0.5 * (az_old[i] + az_new[i]) * dt;
}

/* rk4 -> restore desde original + avance con derivadas k (sin D2D copies) */
__global__ void integr_rk4_restore_advance_kernel(
    double * __restrict__ px, double * __restrict__ py, double * __restrict__ pz,
    double * __restrict__ vx, double * __restrict__ vy, double * __restrict__ vz,
    double * __restrict__ ox, double * __restrict__ oy, double * __restrict__ oz,
    double * __restrict__ ovx, double * __restrict__ ovy, double * __restrict__ ovz,
    double * __restrict__ kvx, double * __restrict__ kvy, double * __restrict__ kvz,
    double * __restrict__ kax, double * __restrict__ kay, double * __restrict__ kaz,
    index_t n, double scale){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    px[i] = ox[i] + kvx[i] * scale;
    py[i] = oy[i] + kvy[i] * scale;
    pz[i] = oz[i] + kvz[i] * scale;
    vx[i] = ovx[i] + kax[i] * scale;
    vy[i] = ovy[i] + kay[i] * scale;
    vz[i] = ovz[i] + kaz[i] * scale;
}

/* rk4: combinacion final con pesos 1,2,2,1 */
__global__ void integr_rk4_combine_kernel(double * __restrict__ px,
                                          double * __restrict__ py,
                                          double * __restrict__ pz,
                                          double * __restrict__ vx,
                                          double * __restrict__ vy,
                                          double * __restrict__ vz,
                                          double * __restrict__ ox,
                                          double * __restrict__ oy,
                                          double * __restrict__ oz,
                                          double * __restrict__ ovx,
                                          double * __restrict__ ovy,
                                          double * __restrict__ ovz,
                                          double * __restrict__ k1ax,
                                          double * __restrict__ k1ay,
                                          double * __restrict__ k1az,
                                          double * __restrict__ k2ax,
                                          double * __restrict__ k2ay,
                                          double * __restrict__ k2az,
                                          double * __restrict__ k3ax,
                                          double * __restrict__ k3ay,
                                          double * __restrict__ k3az,
                                          double * __restrict__ k4ax,
                                          double * __restrict__ k4ay,
                                          double * __restrict__ k4az,
                                          double * __restrict__ k1vx,
                                          double * __restrict__ k1vy,
                                          double * __restrict__ k1vz,
                                          double * __restrict__ k2vx,
                                          double * __restrict__ k2vy,
                                          double * __restrict__ k2vz,
                                          double * __restrict__ k3vx,
                                          double * __restrict__ k3vy,
                                          double * __restrict__ k3vz,
                                          double * __restrict__ k4vx,
                                          double * __restrict__ k4vy,
                                          double * __restrict__ k4vz,
                                          index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    double s6 = dt / 6.0;
    /* x = original + (dt/6)(kv1 + 2*kv2 + 2*kv3 + kv4) */
    px[i] = ox[i] + s6 * (k1vx[i] + 2.0*k2vx[i] + 2.0*k3vx[i] + k4vx[i]);
    py[i] = oy[i] + s6 * (k1vy[i] + 2.0*k2vy[i] + 2.0*k3vy[i] + k4vy[i]);
    pz[i] = oz[i] + s6 * (k1vz[i] + 2.0*k2vz[i] + 2.0*k3vz[i] + k4vz[i]);
    /* v = original + (dt/6)(ka1 + 2*ka2 + 2*ka3 + ka4) */
    vx[i] = ovx[i] + s6 * (k1ax[i] + 2.0*k2ax[i] + 2.0*k3ax[i] + k4ax[i]);
    vy[i] = ovy[i] + s6 * (k1ay[i] + 2.0*k2ay[i] + 2.0*k3ay[i] + k4ay[i]);
    vz[i] = ovz[i] + s6 * (k1az[i] + 2.0*k2az[i] + 2.0*k3az[i] + k4az[i]);
}

/* leapfrog -> v += a*scale (kick) */
__global__ void integr_kick_kernel(double * __restrict__ vx,
                                   double * __restrict__ vy,
                                   double * __restrict__ vz,
                                   double * __restrict__ ax,
                                   double * __restrict__ ay,
                                   double * __restrict__ az,
                                   index_t n, double scale){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    vx[i] += ax[i] * scale;
    vy[i] += ay[i] * scale;
    vz[i] += az[i] * scale;
}

/* leapfrog -> x += v*dt (drift) */
__global__ void integr_drift_kernel(double * __restrict__ px,
                                    double * __restrict__ py,
                                    double * __restrict__ pz,
                                    double * __restrict__ vx,
                                    double * __restrict__ vy,
                                    double * __restrict__ vz,
                                    index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}

int forces_integrate(Universe *u, real dt, index_t steps, IntegratorType integrator_type){
    index_t n = u->n;
    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int rc = gpu_ctx_init(n);
    if(rc) return rc;

    rc = gpu_ctx_upload(u);
    if(rc) return rc;

    gpu_ctx_set_constants(G, SOFTENING);

    switch(integrator_type){

    case INTEGRATOR_EULER: // euler explicito
        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_euler_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, dt);
        }
        break;

    case INTEGRATOR_EULER_SEMIIMPLICIT: //euler semi implicito
        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_semiimplicit_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, dt);
        }
        break;

    case INTEGRATOR_VERLET: //velocity verlet
    {
        double *d_a_old_x, *d_a_old_y, *d_a_old_z;
        CUDA_CHECK(cudaMalloc(&d_a_old_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_a_old_y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_a_old_z, n * sizeof(double)));

        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;

            /*swap pointers en vez de copiar: d_a_old apunta al buffer con a(t)*/
            double *tmp;
            tmp = d_a_old_x; d_a_old_x = gpu_ctx.d_ax; gpu_ctx.d_ax = tmp;
            tmp = d_a_old_y; d_a_old_y = gpu_ctx.d_ay; gpu_ctx.d_ay = tmp;
            tmp = d_a_old_z; d_a_old_z = gpu_ctx.d_az; gpu_ctx.d_az = tmp;

            /*pos usa a(t) que ahora esta en d_a_old*/
            integr_verlet_pos_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_a_old_x, d_a_old_y, d_a_old_z, n, dt);

            rc = launch_forces(blocks, n);
            if(rc) return rc;

            integr_verlet_vel_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_a_old_x, d_a_old_y, d_a_old_z,
                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, dt);
        }

        cudaFree(d_a_old_x);
        cudaFree(d_a_old_y);
        cudaFree(d_a_old_z);
        break;
    }

    case INTEGRATOR_RK4: //RK4 (runge-kutta 4)
    {
        /* buffers para estado original del paso */
        double *d_ox, *d_oy, *d_oz, *d_ovx, *d_ovy, *d_ovz;
        CUDA_CHECK(cudaMalloc(&d_ox,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_oy,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_oz,  n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ovx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ovy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_ovz, n * sizeof(double)));

        /* buffers para derivadas k1..k4 de velocidad (kv) y aceleracion (ka) */
        double *d_k1vx, *d_k1vy, *d_k1vz, *d_k1ax, *d_k1ay, *d_k1az;
        double *d_k2vx, *d_k2vy, *d_k2vz, *d_k2ax, *d_k2ay, *d_k2az;
        double *d_k3vx, *d_k3vy, *d_k3vz, *d_k3ax, *d_k3ay, *d_k3az;
        double *d_k4vx, *d_k4vy, *d_k4vz, *d_k4ax, *d_k4ay, *d_k4az;
        CUDA_CHECK(cudaMalloc(&d_k1vx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1vy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1vz, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1ax, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1ay, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k1az, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2vx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2vy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2vz, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2ax, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2ay, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k2az, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3vx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3vy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3vz, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3ax, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3ay, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k3az, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4vx, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4vy, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4vz, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4ax, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4ay, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_k4az, n * sizeof(double)));

        for(index_t step = 0; step < steps; step++){
            CUDA_CHECK(cudaMemcpy(d_ox,  gpu_ctx.d_px, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_oy,  gpu_ctx.d_py, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_oz,  gpu_ctx.d_pz, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ovx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ovy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_ovz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToDevice));

            /*k1 -> fuerza en estado actual */
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            /* kv1 = v(t), ka1 = a(t) */
            CUDA_CHECK(cudaMemcpy(d_k1vx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k1vy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k1vz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k1ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k1ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k1az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToDevice));

            /*k2 -> restore + avance 0.5·dt·k1 (sin D2D copies) */
            integr_rk4_restore_advance_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_ox, d_oy, d_oz, d_ovx, d_ovy, d_ovz,
                d_k1vx, d_k1vy, d_k1vz, d_k1ax, d_k1ay, d_k1az,
                n, 0.5 * dt);
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            CUDA_CHECK(cudaMemcpy(d_k2vx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k2vy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k2vz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k2ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k2ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k2az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToDevice));

            /*k3 -> restore + avance 0.5·dt·k2 (sin D2D copies) */
            integr_rk4_restore_advance_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_ox, d_oy, d_oz, d_ovx, d_ovy, d_ovz,
                d_k2vx, d_k2vy, d_k2vz, d_k2ax, d_k2ay, d_k2az,
                n, 0.5 * dt);
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            CUDA_CHECK(cudaMemcpy(d_k3vx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k3vy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k3vz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k3ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k3ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k3az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToDevice));

            /*k4: restore + avance dt·k3 (sin D2D copies) */
            integr_rk4_restore_advance_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_ox, d_oy, d_oz, d_ovx, d_ovy, d_ovz,
                d_k3vx, d_k3vy, d_k3vz, d_k3ax, d_k3ay, d_k3az,
                n, dt);
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            CUDA_CHECK(cudaMemcpy(d_k4vx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k4vy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k4vz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k4ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k4ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_k4az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToDevice));

            /*combinacion final estado*/
            integr_rk4_combine_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                d_ox, d_oy, d_oz, d_ovx, d_ovy, d_ovz,
                d_k1ax, d_k1ay, d_k1az,
                d_k2ax, d_k2ay, d_k2az,
                d_k3ax, d_k3ay, d_k3az,
                d_k4ax, d_k4ay, d_k4az,
                d_k1vx, d_k1vy, d_k1vz,
                d_k2vx, d_k2vy, d_k2vz,
                d_k3vx, d_k3vy, d_k3vz,
                d_k4vx, d_k4vy, d_k4vz,
                n, dt);
        }

        cudaFree(d_ox); cudaFree(d_oy); cudaFree(d_oz);
        cudaFree(d_ovx); cudaFree(d_ovy); cudaFree(d_ovz);
        cudaFree(d_k1vx); cudaFree(d_k1vy); cudaFree(d_k1vz);
        cudaFree(d_k1ax); cudaFree(d_k1ay); cudaFree(d_k1az);
        cudaFree(d_k2vx); cudaFree(d_k2vy); cudaFree(d_k2vz);
        cudaFree(d_k2ax); cudaFree(d_k2ay); cudaFree(d_k2az);
        cudaFree(d_k3vx); cudaFree(d_k3vy); cudaFree(d_k3vz);
        cudaFree(d_k3ax); cudaFree(d_k3ay); cudaFree(d_k3az);
        cudaFree(d_k4vx); cudaFree(d_k4vy); cudaFree(d_k4vz);
        cudaFree(d_k4ax); cudaFree(d_k4ay); cudaFree(d_k4az);
        break;
    }

    case INTEGRATOR_LEAPFROG: //leapfrog
        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_kick_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, 0.5 * dt);
            integr_drift_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz, n, dt);
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_kick_kernel<<<blocks, BLOCK_SIZE>>>(
                gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, 0.5 * dt);
        }
        break;

    default:
        return 1;
    }

    rc = gpu_ctx_download(u);
    return rc;
}
