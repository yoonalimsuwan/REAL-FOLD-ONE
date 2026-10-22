"""
=============================================================================
SESI CUDA KERNELS — Production Fused Physics Kernels
=============================================================================
Load via: torch.utils.cpp_extension.load_inline(...)
=============================================================================
"""

CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// ===========================================================================
// C^∞ Surrogate: soft_abs = sqrt(x² + ε²)
// ===========================================================================
__global__ void soft_abs_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = sqrtf(v * v + eps * eps);
    }
}

__global__ void soft_abs_bwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int n, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        float denom = sqrtf(v * v + eps * eps);
        grad_in[idx] = grad_out[idx] * v / (denom + 1e-30f);
    }
}

// ===========================================================================
// C^∞ Surrogate: soft_floor = floor + softplus(x - floor, β)
// ===========================================================================
__device__ __forceinline__ float softplus_f(float x, float beta) {
    float bx = x * beta;
    return fmaxf(bx, 0.0f) + log1pf(expf(-fabsf(bx))) / beta;
}

__global__ void soft_floor_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int n, float floor_val, float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = floor_val + softplus_f(x[idx] - floor_val, beta);
    }
}

__global__ void soft_floor_bwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_in,
    int n, float floor_val, float beta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sig = 1.0f / (1.0f + expf(-beta * (x[idx] - floor_val)));
        grad_in[idx] = grad_out[idx] * sig;
    }
}

// ===========================================================================
// 3D Laplacian — 7-point stencil, periodic BC
// ===========================================================================
__global__ void laplacian_3d_kernel(
    const float* __restrict__ u,
    float* __restrict__ out,
    int nx, int ny, int nz,
    float inv_dx2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int ip1 = (i + 1) % nx, im1 = (i - 1 + nx) % nx;
    int jp1 = (j + 1) % ny, jm1 = (j - 1 + ny) % ny;
    int kp1 = (k + 1) % nz, km1 = (k - 1 + nz) % nz;

    float uc  = u[idx];
    float uxp = u[ip1 * ny * nz + j * nz + k];
    float uxm = u[im1 * ny * nz + j * nz + k];
    float uyp = u[i * ny * nz + jp1 * nz + k];
    float uym = u[i * ny * nz + jm1 * nz + k];
    float uzp = u[i * ny * nz + j * nz + kp1];
    float uzm = u[i * ny * nz + j * nz + km1];

    out[idx] = (uxp + uxm + uyp + uym + uzp + uzm - 6.0f * uc) * inv_dx2;
}

// ===========================================================================
// Fused NS Step: advection + diffusion + forcing
// ===========================================================================
__global__ void fused_ns_step_kernel(
    const float* __restrict__ u,     // [3, nx, ny, nz]
    const float* __restrict__ f,     // [3, nx, ny, nz]
    float* __restrict__ u_star,
    int nx, int ny, int nz,
    float dt, float nu, float inv_dx2, float inv_2dx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int ip1 = (i + 1) % nx, im1 = (i - 1 + nx) % nx;
    int jp1 = (j + 1) % ny, jm1 = (j - 1 + ny) % ny;
    int kp1 = (k + 1) % nz, km1 = (k - 1 + nz) % nz;

    int idx_xp = ip1 * ny * nz + j * nz + k;
    int idx_xm = im1 * ny * nz + j * nz + k;
    int idx_yp = i * ny * nz + jp1 * nz + k;
    int idx_ym = i * ny * nz + jm1 * nz + k;
    int idx_zp = i * ny * nz + j * nz + kp1;
    int idx_zm = i * ny * nz + j * nz + km1;

    // Process all 3 velocity components
    for (int c = 0; c < 3; c++) {
        int base = c * total;
        float uc  = u[base + idx];
        float uxp = u[base + idx_xp];
        float uxm = u[base + idx_xm];
        float uyp = u[base + idx_yp];
        float uym = u[base + idx_ym];
        float uzp = u[base + idx_zp];
        float uzm = u[base + idx_zm];

        // Velocity components at center (for advection)
        float vx = u[idx];           // component 0
        float vy = u[total + idx];   // component 1
        float vz = u[2 * total + idx]; // component 2

        // Gradient of component c
        float dux = (uxp - uxm) * inv_2dx;
        float duy = (uyp - uym) * inv_2dx;
        float duz = (uzp - uzm) * inv_2dx;

        // Advection
        float adv = -(vx * dux + vy * duy + vz * duz);

        // Diffusion
        float lap = (uxp + uxm + uyp + uym + uzp + uzm - 6.0f * uc) * inv_dx2;

        // Forcing
        float fc = f[base + idx];

        u_star[base + idx] = uc + dt * (adv + nu * lap + fc);
    }
}

// ===========================================================================
// Pressure Jacobi iteration
// ===========================================================================
__global__ void pressure_jacobi_kernel(
    const float* __restrict__ p,
    const float* __restrict__ rhs,
    float* __restrict__ p_new,
    int nx, int ny, int nz,
    float dx2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int ip1 = (i + 1) % nx, im1 = (i - 1 + nx) % nx;
    int jp1 = (j + 1) % ny, jm1 = (j - 1 + ny) % ny;
    int kp1 = (k + 1) % nz, km1 = (k - 1 + nz) % nz;

    float pxp = p[ip1 * ny * nz + j * nz + k];
    float pxm = p[im1 * ny * nz + j * nz + k];
    float pyp = p[i * ny * nz + jp1 * nz + k];
    float pym = p[i * ny * nz + jm1 * nz + k];
    float pzp = p[i * ny * nz + j * nz + kp1];
    float pzm = p[i * ny * nz + j * nz + km1];

    p_new[idx] = (pxp + pxm + pyp + pym + pzp + pzm - dx2 * rhs[idx]) / 6.0f;
}

// ===========================================================================
// Chorin projection: u_next = u_star - dt · ∇p
// ===========================================================================
__global__ void chorin_projection_kernel(
    const float* __restrict__ u_star,
    const float* __restrict__ p,
    float* __restrict__ u_next,
    int nx, int ny, int nz,
    float dt, float inv_2dx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (nz * ny);

    int ip1 = (i + 1) % nx, im1 = (i - 1 + nx) % nx;
    int jp1 = (j + 1) % ny, jm1 = (j - 1 + ny) % ny;
    int kp1 = (k + 1) % nz, km1 = (k - 1 + nz) % nz;

    int idx_xp = ip1 * ny * nz + j * nz + k;
    int idx_xm = im1 * ny * nz + j * nz + k;
    int idx_yp = i * ny * nz + jp1 * nz + k;
    int idx_ym = i * ny * nz + jm1 * nz + k;
    int idx_zp = i * ny * nz + j * nz + kp1;
    int idx_zm = i * ny * nz + j * nz + km1;

    float gpx = (p[idx_xp] - p[idx_xm]) * inv_2dx;
    float gpy = (p[idx_yp] - p[idx_ym]) * inv_2dx;
    float gpz = (p[idx_zp] - p[idx_zm]) * inv_2dx;

    u_next[idx]             = u_star[idx]             - dt * gpx;
    u_next[total + idx]     = u_star[total + idx]     - dt * gpy;
    u_next[2 * total + idx] = u_star[2 * total + idx] - dt * gpz;
}

// ===========================================================================
// C++ bindings
// ===========================================================================

torch::Tensor soft_abs_cuda(torch::Tensor x, float eps) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    soft_abs_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n, eps);
    return out;
}

torch::Tensor soft_abs_bwd_cuda(torch::Tensor x, torch::Tensor grad_out, float eps) {
    auto grad_in = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    soft_abs_bwd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), grad_out.data_ptr<float>(),
        grad_in.data_ptr<float>(), n, eps);
    return grad_in;
}

torch::Tensor soft_floor_cuda(torch::Tensor x, float floor_val, float beta) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    soft_floor_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), n, floor_val, beta);
    return out;
}

torch::Tensor soft_floor_bwd_cuda(torch::Tensor x, torch::Tensor grad_out,
                                   float floor_val, float beta) {
    auto grad_in = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    soft_floor_bwd_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), grad_out.data_ptr<float>(),
        grad_in.data_ptr<float>(), n, floor_val, beta);
    return grad_in;
}

torch::Tensor laplacian_3d_cuda(torch::Tensor u, float dx) {
    // u: (D, H, W) single channel
    auto out = torch::empty_like(u);
    int nx = u.size(0), ny = u.size(1), nz = u.size(2);
    int total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float inv_dx2 = 1.0f / (dx * dx);
    laplacian_3d_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(), out.data_ptr<float>(),
        nx, ny, nz, inv_dx2);
    return out;
}

torch::Tensor fused_ns_step_cuda(
    torch::Tensor u, torch::Tensor f,
    float dx, float dt, float nu
) {
    // u, f: (3, D, H, W)
    auto u_star = torch::empty_like(u);
    int nx = u.size(1), ny = u.size(2), nz = u.size(3);
    int total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float inv_dx2 = 1.0f / (dx * dx);
    float inv_2dx = 1.0f / (2.0f * dx);
    fused_ns_step_kernel<<<blocks, threads>>>(
        u.data_ptr<float>(), f.data_ptr<float>(), u_star.data_ptr<float>(),
        nx, ny, nz, dt, nu, inv_dx2, inv_2dx);
    return u_star;
}

torch::Tensor pressure_jacobi_cuda(
    torch::Tensor p, torch::Tensor rhs,
    float dx, int n_iter
) {
    int nx = p.size(0), ny = p.size(1), nz = p.size(2);
    int total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float dx2 = dx * dx;

    auto p_cur = p.contiguous();
    auto p_next = torch::empty_like(p_cur);

    for (int iter = 0; iter < n_iter; iter++) {
        pressure_jacobi_kernel<<<blocks, threads>>>(
            p_cur.data_ptr<float>(), rhs.data_ptr<float>(),
            p_next.data_ptr<float>(), nx, ny, nz, dx2);
        std::swap(p_cur, p_next);
    }
    return p_cur;
}

torch::Tensor chorin_projection_cuda(
    torch::Tensor u_star, torch::Tensor p,
    float dx, float dt
) {
    auto u_next = torch::empty_like(u_star);
    int nx = u_star.size(1), ny = u_star.size(2), nz = u_star.size(3);
    int total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    float inv_2dx = 1.0f / (2.0f * dx);
    chorin_projection_kernel<<<blocks, threads>>>(
        u_star.data_ptr<float>(), p.data_ptr<float>(),
        u_next.data_ptr<float>(), nx, ny, nz, dt, inv_2dx);
    return u_next;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_abs", &soft_abs_cuda, "soft_abs (CUDA)");
    m.def("soft_abs_bwd", &soft_abs_bwd_cuda, "soft_abs backward (CUDA)");
    m.def("soft_floor", &soft_floor_cuda, "soft_floor (CUDA)");
    m.def("soft_floor_bwd", &soft_floor_bwd_cuda, "soft_floor backward (CUDA)");
    m.def("laplacian_3d", &laplacian_3d_cuda, "3D Laplacian (CUDA)");
    m.def("fused_ns_step", &fused_ns_step_cuda, "Fused NS step (CUDA)");
    m.def("pressure_jacobi", &pressure_jacobi_cuda, "Pressure Jacobi (CUDA)");
    m.def("chorin_projection", &chorin_projection_cuda, "Chorin projection (CUDA)");
}
"""

# Load at import time (cached by torch)
def load_cuda_kernels():
    from torch.utils.cpp_extension import load_inline
    return load_inline(
        name="sesi_cuda_kernels",
        cpp_sources="",
        cuda_sources=CUDA_SOURCE,
        functions=[
            "soft_abs", "soft_abs_bwd",
            "soft_floor", "soft_floor_bwd",
            "laplacian_3d", "fused_ns_step",
            "pressure_jacobi", "chorin_projection",
        ],
        verbose=False,
    )
