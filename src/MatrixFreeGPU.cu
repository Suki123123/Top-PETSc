/**
 * MatrixFreeGPU.cu - GPU实现的Matrix-Free矩阵-向量乘积
 *
 * 将单元级计算移植到GPU，避免GPU↔CPU数据传输
 */

#include <cuda_runtime.h>
#include <petsc.h>
#include <petscvec.h>
#include <mpi.h>

// 单元刚度矩阵大小
#define EDOF 24
#define NEN 8

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            PetscPrintf(PETSC_COMM_WORLD, "CUDA error at %s:%d: %s\n", \
                       __FILE__, __LINE__, cudaGetErrorString(err)); \
            return PETSC_ERR_LIB; \
        } \
    } while(0)

/**
 * GPU内核：计算单元贡献并累加到局部向量
 *
 * 每个线程处理一个单元
 * 使用原子操作累加到局部向量（因为节点被多个单元共享）
 */
__global__ void MatMult_Kernel(
    PetscScalar* __restrict__ y_local,     // 输出向量（局部，含ghost）
    const PetscScalar* __restrict__ x_local, // 输入向量（局部，含ghost）
    const PetscScalar* __restrict__ xPhys, // 密度向量
    const PetscScalar* __restrict__ KE,    // 单元刚度矩阵 (24x24)
    const PetscInt* __restrict__ necon,    // 单元连接性 (nel x 8)
    PetscInt nel,                          // 单元数
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= nel) return;

    // 获取单元节点
    PetscInt nodes[NEN];
    for (int j = 0; j < NEN; j++) {
        nodes[j] = necon[elem * NEN + j];
    }

    // 提取单元位移向量
    PetscScalar xe[EDOF];
    for (int j = 0; j < NEN; j++) {
        for (int k = 0; k < 3; k++) {
            xe[j * 3 + k] = x_local[3 * nodes[j] + k];
        }
    }

    // 计算材料系数
    PetscScalar dens_val = xPhys[elem];
    PetscScalar dens = Emin + pow(dens_val, penal) * (Emax - Emin);

    // 计算 ye = K_e * xe
    PetscScalar ye[EDOF];
    for (int j = 0; j < EDOF; j++) {
        ye[j] = 0.0;
        for (int k = 0; k < EDOF; k++) {
            ye[j] += KE[j * EDOF + k] * xe[k];
        }
        ye[j] *= dens;
    }

    // 使用原子操作累加到局部向量（因为节点被多个单元共享）
    for (int j = 0; j < NEN; j++) {
        for (int k = 0; k < 3; k++) {
            atomicAdd(&y_local[3 * nodes[j] + k], ye[j * 3 + k]);
        }
    }
}

/**
 * GPU内核：应用Dirichlet边界条件
 * y = N * y_computed + (1 - N) * x
 */
__global__ void ApplyBC_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ N,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar n_val = N[i];
    y[i] = n_val * y[i] + (1.0 - n_val) * x[i];
}

/**
 * GPU内核：计算对角线元素（用于Jacobi预条件器）
 */
__global__ void GetDiagonal_Kernel(
    PetscScalar* __restrict__ diag,
    const PetscScalar* __restrict__ xPhys,
    const PetscScalar* __restrict__ KE,
    const PetscInt* __restrict__ necon,
    PetscInt nel,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= nel) return;

    // 计算材料系数
    PetscScalar dens_val = xPhys[elem];
    PetscScalar dens = Emin + pow(dens_val, penal) * (Emax - Emin);

    // 获取单元节点
    PetscInt nodes[NEN];
    for (int j = 0; j < NEN; j++) {
        nodes[j] = necon[elem * NEN + j];
    }

    // 累加对角线贡献
    for (int j = 0; j < NEN; j++) {
        for (int k = 0; k < 3; k++) {
            int local_dof = j * 3 + k;
            int global_dof = 3 * nodes[j] + k;
            atomicAdd(&diag[global_dof], dens * KE[local_dof * EDOF + local_dof]);
        }
    }
}

/**
 * GPU内核：计算目标函数和敏感度
 * 每个线程处理一个单元
 */
__global__ void ComputeSensitivity_Kernel(
    PetscScalar* __restrict__ fx_partial,  // 部分目标函数值（每个单元一个）
    PetscScalar* __restrict__ dfdx,        // 敏感度向量
    const PetscScalar* __restrict__ u_local, // 位移向量（局部，含ghost）
    const PetscScalar* __restrict__ xPhys, // 密度向量
    const PetscScalar* __restrict__ KE,    // 单元刚度矩阵
    const PetscInt* __restrict__ necon,    // 单元连接性
    PetscInt nel,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    int elem = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem >= nel) return;

    // 获取单元节点
    PetscInt nodes[NEN];
    for (int j = 0; j < NEN; j++) {
        nodes[j] = necon[elem * NEN + j];
    }

    // 提取单元位移向量
    PetscScalar ue[EDOF];
    for (int j = 0; j < NEN; j++) {
        for (int k = 0; k < 3; k++) {
            ue[j * 3 + k] = u_local[3 * nodes[j] + k];
        }
    }

    // 计算 uKu = u^T * K_e * u
    PetscScalar uKu = 0.0;
    for (int k = 0; k < EDOF; k++) {
        for (int h = 0; h < EDOF; h++) {
            uKu += ue[k] * KE[k * EDOF + h] * ue[h];
        }
    }

    // 获取密度
    PetscScalar x = xPhys[elem];

    // 计算目标函数贡献
    PetscScalar dens = Emin + pow(x, penal) * (Emax - Emin);
    fx_partial[elem] = dens * uKu;

    // 计算敏感度
    dfdx[elem] = -penal * pow(x, penal - 1.0) * (Emax - Emin) * uKu;
}

// ============================================================================
// C接口函数（供PETSc调用）
// ============================================================================

// GPU资源结构
typedef struct {
    PetscScalar* d_KE;      // 设备端单元刚度矩阵
    PetscInt*    d_necon;   // 设备端单元连接性
    PetscInt     nel;       // 单元数
    PetscInt     n_local;   // 本地节点数（含ghost）
    PetscInt     n_owned;   // 本地拥有的DOF数
    cudaStream_t stream;    // CUDA流
} GPUResources;

extern "C" {

/**
 * 初始化GPU资源
 */
PetscErrorCode MatrixFreeGPU_Init(
    GPUResources** resources,
    const PetscScalar* KE,
    const PetscInt* necon,
    PetscInt nel,
    PetscInt n_local,
    PetscInt n_owned)
{
    GPUResources* res = new GPUResources;

    res->nel = nel;
    res->n_local = n_local;
    res->n_owned = n_owned;

    // 创建CUDA流
    CUDA_CHECK(cudaStreamCreate(&res->stream));

    // 分配并复制单元刚度矩阵到GPU
    CUDA_CHECK(cudaMalloc(&res->d_KE, EDOF * EDOF * sizeof(PetscScalar)));
    CUDA_CHECK(cudaMemcpy(res->d_KE, KE, EDOF * EDOF * sizeof(PetscScalar),
                          cudaMemcpyHostToDevice));

    // 分配并复制单元连接性到GPU
    CUDA_CHECK(cudaMalloc(&res->d_necon, nel * NEN * sizeof(PetscInt)));
    CUDA_CHECK(cudaMemcpy(res->d_necon, necon, nel * NEN * sizeof(PetscInt),
                          cudaMemcpyHostToDevice));

    *resources = res;
    return 0;
}

/**
 * 释放GPU资源
 */
PetscErrorCode MatrixFreeGPU_Destroy(GPUResources* res)
{
    if (res) {
        cudaFree(res->d_KE);
        cudaFree(res->d_necon);
        cudaStreamDestroy(res->stream);
        delete res;
    }
    return 0;
}

/**
 * GPU版本的单元循环
 *
 * 只执行单元级计算，不处理边界条件
 * 边界条件由调用者处理
 *
 * 输入：
 *   - d_x_local: 局部向量（含ghost节点），已在GPU上
 *   - d_xPhys: 密度向量，已在GPU上
 *
 * 输出：
 *   - d_y_local: 局部输出向量，已在GPU上
 */
PetscErrorCode MatrixFreeGPU_MatMult(
    GPUResources* res,
    PetscScalar* d_y_local,        // 输出（局部，GPU）
    const PetscScalar* d_x_local,  // 输入（局部含ghost，GPU）
    const PetscScalar* d_x_global, // 未使用
    const PetscScalar* d_xPhys,    // 密度（GPU）
    const PetscScalar* d_N,        // 未使用
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    (void)d_x_global;  // 未使用
    (void)d_N;         // 未使用

    // 清零局部输出向量
    CUDA_CHECK(cudaMemsetAsync(d_y_local, 0, res->n_local * sizeof(PetscScalar), res->stream));

    // 启动MatMult内核
    int blockSize = 256;
    int numBlocks = (res->nel + blockSize - 1) / blockSize;

    MatMult_Kernel<<<numBlocks, blockSize, 0, res->stream>>>(
        d_y_local, d_x_local, d_xPhys, res->d_KE, res->d_necon,
        res->nel, Emin, Emax, penal);

    // 同步
    CUDA_CHECK(cudaStreamSynchronize(res->stream));

    return 0;
}

/**
 * GPU版本的边界条件应用
 */
PetscErrorCode MatrixFreeGPU_ApplyBC(
    GPUResources* res,
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_N,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    ApplyBC_Kernel<<<numBlocks, blockSize, 0, res->stream>>>(
        d_y, d_x, d_N, n);

    CUDA_CHECK(cudaStreamSynchronize(res->stream));

    return 0;
}

/**
 * GPU版本的GetDiagonal
 */
PetscErrorCode MatrixFreeGPU_GetDiagonal(
    GPUResources* res,
    PetscScalar* d_diag,
    const PetscScalar* d_xPhys,
    const PetscScalar* d_N,
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    // 清零对角线向量 - 使用n_local（包含ghost节点）因为necon使用局部索引
    CUDA_CHECK(cudaMemsetAsync(d_diag, 0, res->n_local * sizeof(PetscScalar), res->stream));

    // 启动GetDiagonal内核
    int blockSize = 256;
    int numBlocks = (res->nel + blockSize - 1) / blockSize;

    GetDiagonal_Kernel<<<numBlocks, blockSize, 0, res->stream>>>(
        d_diag, d_xPhys, res->d_KE, res->d_necon,
        res->nel, Emin, Emax, penal);

    // 同步
    CUDA_CHECK(cudaStreamSynchronize(res->stream));

    return 0;
}

/**
 * GPU版本的敏感度计算
 * 计算目标函数和敏感度
 */
PetscErrorCode MatrixFreeGPU_ComputeSensitivity(
    GPUResources* res,
    PetscScalar* d_fx_partial,     // 输出：部分目标函数值（每个单元一个）
    PetscScalar* d_dfdx,           // 输出：敏感度向量
    const PetscScalar* d_u_local,  // 输入：位移向量（局部，含ghost）
    const PetscScalar* d_xPhys,    // 输入：密度向量
    PetscScalar Emin,
    PetscScalar Emax,
    PetscScalar penal)
{
    // 启动敏感度计算内核
    int blockSize = 256;
    int numBlocks = (res->nel + blockSize - 1) / blockSize;

    ComputeSensitivity_Kernel<<<numBlocks, blockSize, 0, res->stream>>>(
        d_fx_partial, d_dfdx, d_u_local, d_xPhys, res->d_KE, res->d_necon,
        res->nel, Emin, Emax, penal);

    // 同步
    CUDA_CHECK(cudaStreamSynchronize(res->stream));

    return 0;
}

/**
 * GPU归约内核：计算数组元素之和
 * 使用共享内存进行块内归约
 */
__global__ void ReduceSum_Kernel(
    PetscScalar* __restrict__ output,
    const PetscScalar* __restrict__ input,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // 每个线程加载两个元素到共享内存
    PetscScalar sum = 0.0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // 块内归约
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写入结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU版本的数组求和（归约）
 */
PetscErrorCode MatrixFreeGPU_ReduceSum(
    PetscScalar* result,
    const PetscScalar* d_array,
    PetscInt n)
{
    if (n <= 0) {
        *result = 0.0;
        return 0;
    }

    int blockSize = 256;
    int numBlocks = (n + blockSize * 2 - 1) / (blockSize * 2);

    // 分配临时GPU内存
    PetscScalar* d_partial;
    CUDA_CHECK(cudaMalloc(&d_partial, numBlocks * sizeof(PetscScalar)));

    // 第一次归约
    ReduceSum_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial, d_array, n);

    // 如果还有多个块，继续归约
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        PetscScalar* d_partial2;
        CUDA_CHECK(cudaMalloc(&d_partial2, newNumBlocks * sizeof(PetscScalar)));

        ReduceSum_Kernel<<<newNumBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
            d_partial2, d_partial, numBlocks);

        CUDA_CHECK(cudaFree(d_partial));
        d_partial = d_partial2;
        numBlocks = newNumBlocks;
    }

    // 复制结果到CPU
    CUDA_CHECK(cudaMemcpy(result, d_partial, sizeof(PetscScalar), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_partial));

    return 0;
}

/**
 * 获取单元数量
 */
PetscInt MatrixFreeGPU_GetNel(GPUResources* res)
{
    return res->nel;
}

// ============================================================================
// Filter相关GPU内核
// ============================================================================

/**
 * GPU内核：Heaviside投影滤波器
 * y[i] = (tanh(beta*eta) + tanh(beta*(x[i]-eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))
 */
__global__ void HeavisideFilter_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    y[i] = (tanh(beta * eta) + tanh(beta * (xi - eta))) /
           (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
}

/**
 * GPU内核：Heaviside投影滤波器的链式法则
 * y[i] = beta * (1 - tanh(beta*(x[i]-eta))^2) / (tanh(beta*eta) + tanh(beta*(1-eta)))
 */
__global__ void ChainruleHeavisideFilter_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar t = tanh(beta * (xi - eta));
    y[i] = beta * (1.0 - t * t) / (tanh(beta * eta) + tanh(beta * (1.0 - eta)));
}

/**
 * GPU内核：计算MND（Measure of Non-Discreteness）
 * 使用块级归约
 */
__global__ void GetMND_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ x,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    PetscScalar val = 0.0;
    if (i < n) {
        PetscScalar xi = x[i];
        val = 4.0 * xi * (1.0 - xi);
    }
    sdata[tid] = val;
    __syncthreads();

    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写入部分和
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：元素级乘法 y[i] = y[i] * dx[i]
 */
__global__ void ElementwiseMult_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ dx,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = y[i] * dx[i];
}

/**
 * GPU内核：边界检查和修正（用于PDE滤波器）
 */
__global__ void BoundCheck_Kernel(
    PetscScalar* __restrict__ x,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (x[i] < 0.0) {
        x[i] = 0.0;
    }
    if (x[i] > 1.0) {
        x[i] = 1.0;
    }
}

// ============================================================================
// Filter相关C接口函数
// ============================================================================

/**
 * GPU版本的Heaviside投影滤波器
 */
PetscErrorCode FilterGPU_HeavisideFilter(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    HeavisideFilter_Kernel<<<numBlocks, blockSize>>>(d_y, d_x, n, beta, eta);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的Heaviside链式法则
 */
PetscErrorCode FilterGPU_ChainruleHeavisideFilter(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar beta,
    PetscScalar eta)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    ChainruleHeavisideFilter_Kernel<<<numBlocks, blockSize>>>(d_y, d_x, n, beta, eta);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MND计算
 */
PetscErrorCode FilterGPU_GetMND(
    PetscScalar* result,
    const PetscScalar* d_x,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 分配部分和数组
    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    // 第一次归约
    GetMND_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制部分和到CPU并完成归约
    PetscScalar* h_partial_sums = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial_sums[i];
    }

    *result = sum;

    // 清理
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU版本的元素级乘法
 */
PetscErrorCode FilterGPU_ElementwiseMult(
    PetscScalar* d_y,
    const PetscScalar* d_dx,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    ElementwiseMult_Kernel<<<numBlocks, blockSize>>>(d_y, d_dx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的边界检查
 */
PetscErrorCode FilterGPU_BoundCheck(
    PetscScalar* d_x,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    BoundCheck_Kernel<<<numBlocks, blockSize>>>(d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

// ============================================================================
// OC优化器相关GPU内核
// ============================================================================

/**
 * GPU内核：OC更新（给定lambda）
 * x_new = max(0, max(x-move, min(1, min(x+move, x * sqrt(-dfdx / (lambda * dgdx))))))
 */
__global__ void OC_Update_Kernel(
    PetscScalar* __restrict__ x_new,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ dfdx,
    const PetscScalar* __restrict__ dgdx,
    PetscInt n,
    PetscScalar lambda,
    PetscScalar move,
    PetscScalar xmin,
    PetscScalar xmax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar df = dfdx[i];
    PetscScalar dg = dgdx[i];

    // OC更新公式
    PetscScalar Be = -df / (lambda * dg);
    if (Be < 0.0) Be = 0.0;  // 防止负数开方
    PetscScalar x_oc = xi * sqrt(Be);

    // 应用移动限制
    PetscScalar x_lower = max(xmin, xi - move);
    PetscScalar x_upper = min(xmax, xi + move);

    // 限制到[x_lower, x_upper]
    x_new[i] = max(x_lower, min(x_upper, x_oc));
}

/**
 * GPU内核：计算体积（归约）
 */
__global__ void ComputeVolume_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ x,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    PetscScalar val = 0.0;
    if (i < n) {
        val = x[i];
    }
    sdata[tid] = val;
    __syncthreads();

    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写入部分和
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：计算设计变化量（最大绝对差）
 */
__global__ void ComputeChange_Kernel(
    PetscScalar* __restrict__ partial_max,
    const PetscScalar* __restrict__ x_new,
    const PetscScalar* __restrict__ x_old,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    PetscScalar val = 0.0;
    if (i < n) {
        val = fabs(x_new[i] - x_old[i]);
    }
    sdata[tid] = val;
    __syncthreads();

    // 块内归约（取最大值）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // 写入部分最大值
    if (tid == 0) {
        partial_max[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：复制向量
 */
__global__ void VecCopy_Kernel(
    PetscScalar* __restrict__ dst,
    const PetscScalar* __restrict__ src,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = src[i];
}

// ============================================================================
// OC优化器C接口函数
// ============================================================================

/**
 * GPU版本的OC更新（给定lambda）
 */
PetscErrorCode OCGPU_Update(
    PetscScalar* d_x_new,
    const PetscScalar* d_x,
    const PetscScalar* d_dfdx,
    const PetscScalar* d_dgdx,
    PetscInt n,
    PetscScalar lambda,
    PetscScalar move,
    PetscScalar xmin,
    PetscScalar xmax)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    OC_Update_Kernel<<<numBlocks, blockSize>>>(
        d_x_new, d_x, d_dfdx, d_dgdx, n, lambda, move, xmin, xmax);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的体积计算
 */
PetscErrorCode OCGPU_ComputeVolume(
    PetscScalar* result,
    const PetscScalar* d_x,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 分配部分和数组
    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    // 第一次归约
    ComputeVolume_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_x, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制部分和到CPU并完成归约
    PetscScalar* h_partial_sums = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial_sums[i];
    }

    *result = sum;

    // 清理
    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU版本的设计变化量计算
 */
PetscErrorCode OCGPU_ComputeChange(
    PetscScalar* result,
    const PetscScalar* d_x_new,
    const PetscScalar* d_x_old,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 分配部分最大值数组
    PetscScalar* d_partial_max;
    CUDA_CHECK(cudaMalloc(&d_partial_max, numBlocks * sizeof(PetscScalar)));

    // 第一次归约
    ComputeChange_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_max, d_x_new, d_x_old, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 复制部分最大值到CPU并完成归约
    PetscScalar* h_partial_max = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_max, d_partial_max,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar max_val = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        if (h_partial_max[i] > max_val) {
            max_val = h_partial_max[i];
        }
    }

    *result = max_val;

    // 清理
    delete[] h_partial_max;
    cudaFree(d_partial_max);

    return 0;
}

/**
 * GPU版本的向量复制
 */
PetscErrorCode OCGPU_VecCopy(
    PetscScalar* d_dst,
    const PetscScalar* d_src,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    VecCopy_Kernel<<<numBlocks, blockSize>>>(d_dst, d_src, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的OC优化器（带二分法求lambda）
 * 返回设计变化量
 */
PetscErrorCode OCGPU_Optimize(
    PetscScalar* d_x,           // 输入/输出：设计变量
    PetscScalar* d_x_old,       // 工作空间：旧设计变量
    PetscScalar* d_x_new,       // 工作空间：新设计变量
    const PetscScalar* d_dfdx,  // 输入：目标函数敏感度
    const PetscScalar* d_dgdx,  // 输入：约束敏感度
    PetscInt n_local,           // 本地元素数
    PetscInt n_global,          // 全局元素数
    PetscScalar volfrac,        // 目标体积分数
    PetscScalar move,           // 移动限制
    PetscScalar* change)        // 输出：设计变化量
{
    PetscScalar xmin = 0.0;
    PetscScalar xmax = 1.0;

    // 保存旧设计
    OCGPU_VecCopy(d_x_old, d_x, n_local);

    // 二分法求lambda
    PetscScalar l1 = 1e-9;
    PetscScalar l2 = 1e9;
    PetscScalar lmid;

    PetscScalar vol_local, vol_global;
    PetscScalar target_vol = volfrac * n_global;

    int max_iter = 100;
    for (int iter = 0; iter < max_iter; iter++) {
        lmid = 0.5 * (l1 + l2);

        // 使用当前lambda更新设计
        OCGPU_Update(d_x_new, d_x_old, d_dfdx, d_dgdx, n_local, lmid, move, xmin, xmax);

        // 计算体积
        OCGPU_ComputeVolume(&vol_local, d_x_new, n_local);

        // MPI归约得到全局体积
        MPI_Allreduce(&vol_local, &vol_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

        // 更新二分法边界
        if (vol_global > target_vol) {
            l1 = lmid;
        } else {
            l2 = lmid;
        }

        // 检查收敛
        if ((l2 - l1) / (l1 + l2) < 1e-4) {
            break;
        }
    }

    // 计算设计变化量
    PetscScalar change_local, change_global;
    OCGPU_ComputeChange(&change_local, d_x_new, d_x_old, n_local);
    MPI_Allreduce(&change_local, &change_global, 1, MPIU_SCALAR, MPI_MAX, PETSC_COMM_WORLD);

    // 复制新设计到输出
    OCGPU_VecCopy(d_x, d_x_new, n_local);

    *change = change_global;

    return 0;
}

// ============================================================================
// 额外的GPU内核：SetOuterMovelimit和滤波器
// ============================================================================

/**
 * GPU内核：设置移动限制
 * xmin[i] = max(Xmin, x[i] - move)
 * xmax[i] = min(Xmax, x[i] + move)
 */
__global__ void SetOuterMovelimit_Kernel(
    PetscScalar* __restrict__ xmin,
    PetscScalar* __restrict__ xmax,
    const PetscScalar* __restrict__ x,
    PetscInt n,
    PetscScalar Xmin,
    PetscScalar Xmax,
    PetscScalar move)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    xmin[i] = max(Xmin, xi - move);
    xmax[i] = min(Xmax, xi + move);
}

/**
 * GPU内核：向量点除 y[i] = x[i] / z[i]
 */
__global__ void VecPointwiseDivide_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ z,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = x[i] / z[i];
}

/**
 * GPU内核：向量点乘 y[i] = x[i] * z[i]
 */
__global__ void VecPointwiseMultiply_Kernel(
    PetscScalar* __restrict__ y,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ z,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    y[i] = x[i] * z[i];
}

/**
 * GPU版本的SetOuterMovelimit
 */
PetscErrorCode OCGPU_SetOuterMovelimit(
    PetscScalar* d_xmin,
    PetscScalar* d_xmax,
    const PetscScalar* d_x,
    PetscInt n,
    PetscScalar Xmin,
    PetscScalar Xmax,
    PetscScalar move)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    SetOuterMovelimit_Kernel<<<numBlocks, blockSize>>>(
        d_xmin, d_xmax, d_x, n, Xmin, Xmax, move);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的向量点除
 */
PetscErrorCode VecGPU_PointwiseDivide(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_z,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    VecPointwiseDivide_Kernel<<<numBlocks, blockSize>>>(d_y, d_x, d_z, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的向量点乘
 */
PetscErrorCode VecGPU_PointwiseMultiply(
    PetscScalar* d_y,
    const PetscScalar* d_x,
    const PetscScalar* d_z,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    VecPointwiseMultiply_Kernel<<<numBlocks, blockSize>>>(d_y, d_x, d_z, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

// ============================================================================
// MMA优化器相关GPU内核
// ============================================================================

/**
 * GPU内核：MMA初始化渐近线（k < 3时）
 * L[i] = x[i] - asyminit * (xmax[i] - xmin[i])
 * U[i] = x[i] + asyminit * (xmax[i] - xmin[i])
 */
__global__ void MMA_InitAsymptotes_Kernel(
    PetscScalar* __restrict__ L,
    PetscScalar* __restrict__ U,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ xmin,
    const PetscScalar* __restrict__ xmax,
    PetscInt n,
    PetscScalar asyminit)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar diff = xmax[i] - xmin[i];
    L[i] = xi - asyminit * diff;
    U[i] = xi + asyminit * diff;
}

/**
 * GPU内核：MMA更新渐近线（k >= 3时）
 */
__global__ void MMA_UpdateAsymptotes_Kernel(
    PetscScalar* __restrict__ L,
    PetscScalar* __restrict__ U,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ xo1,
    const PetscScalar* __restrict__ xo2,
    const PetscScalar* __restrict__ xmin,
    const PetscScalar* __restrict__ xmax,
    PetscInt n,
    PetscScalar asymdec,
    PetscScalar asyminc)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar x1i = xo1[i];
    PetscScalar x2i = xo2[i];
    PetscScalar Li = L[i];
    PetscScalar Ui = U[i];

    PetscScalar helpvar = (xi - x1i) * (x1i - x2i);
    PetscScalar gamma;
    if (helpvar < 0.0) {
        gamma = asymdec;
    } else if (helpvar > 0.0) {
        gamma = asyminc;
    } else {
        gamma = 1.0;
    }

    Li = xi - gamma * (x1i - Li);
    Ui = xi + gamma * (Ui - x1i);

    // 限制渐近线范围
    PetscScalar xmi = max(1.0e-5, xmax[i] - xmin[i]);
    Li = max(Li, xi - 10.0 * xmi);
    Li = min(Li, xi - 0.01 * xmi);
    Ui = max(Ui, xi + 0.01 * xmi);
    Ui = min(Ui, xi + 10.0 * xmi);

    L[i] = Li;
    U[i] = Ui;
}

/**
 * GPU内核：MMA计算alpha, beta, p0, q0
 */
__global__ void MMA_ComputeAlphaBetaPQ_Kernel(
    PetscScalar* __restrict__ alpha,
    PetscScalar* __restrict__ beta,
    PetscScalar* __restrict__ p0,
    PetscScalar* __restrict__ q0,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    const PetscScalar* __restrict__ xmin,
    const PetscScalar* __restrict__ xmax,
    const PetscScalar* __restrict__ dfdx,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar Li = L[i];
    PetscScalar Ui = U[i];
    PetscScalar df = dfdx[i];

    // 计算alpha和beta
    alpha[i] = max(xmin[i], 0.9 * Li + 0.1 * xi);
    beta[i] = min(xmax[i], 0.9 * Ui + 0.1 * xi);

    // 计算p0和q0
    PetscScalar feps = 1.0e-6;
    PetscScalar dfdxp = max(0.0, df);
    PetscScalar dfdxm = max(0.0, -df);
    PetscScalar UmX = Ui - xi;
    PetscScalar XmL = xi - Li;
    PetscScalar UmL = Ui - Li;

    p0[i] = UmX * UmX * (dfdxp + 0.001 * fabs(df) + 0.5 * feps / UmL);
    q0[i] = XmL * XmL * (dfdxm + 0.001 * fabs(df) + 0.5 * feps / UmL);
}

/**
 * GPU内核：MMA计算pij和qij（约束敏感度）
 */
__global__ void MMA_ComputePijQij_Kernel(
    PetscScalar* __restrict__ pij,
    PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    const PetscScalar* __restrict__ dgdx,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar Li = L[i];
    PetscScalar Ui = U[i];
    PetscScalar dg = dgdx[i];

    PetscScalar UmX = Ui - xi;
    PetscScalar XmL = xi - Li;

    PetscScalar dgdxp = max(0.0, dg);
    PetscScalar dgdxm = max(0.0, -dg);

    pij[i] = UmX * UmX * dgdxp;
    qij[i] = XmL * XmL * dgdxm;
}

/**
 * GPU内核：MMA计算b的部分和
 * b[j] = sum(pij[j][i] / (U[i] - x[i]) + qij[j][i] / (x[i] - L[i]))
 */
__global__ void MMA_ComputeB_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    PetscScalar val = 0.0;
    if (i < n) {
        PetscScalar UmX = U[i] - x[i];
        PetscScalar XmL = x[i] - L[i];
        val = pij[i] / UmX + qij[i] / XmL;
    }
    sdata[tid] = val;
    __syncthreads();

    // 块内归约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：MMA XYZofLAMBDA - 根据lambda计算x
 */
__global__ void MMA_XYZofLAMBDA_Kernel(
    PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ p0,
    const PetscScalar* __restrict__ q0,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ alpha,
    const PetscScalar* __restrict__ beta,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    PetscInt n,
    PetscScalar lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar pjlam = p0[i] + pij[i] * lam;
    PetscScalar qjlam = q0[i] + qij[i] * lam;

    PetscScalar sqp = sqrt(pjlam);
    PetscScalar sqq = sqrt(qjlam);

    PetscScalar xi = (sqp * L[i] + sqq * U[i]) / (sqp + sqq);

    // 限制到[alpha, beta]
    if (xi < alpha[i]) xi = alpha[i];
    if (xi > beta[i]) xi = beta[i];

    x[i] = xi;
}

/**
 * GPU内核：MMA XYZofLAMBDA - 从GPU数组读取lam（避免CPU-GPU通信）
 */
__global__ void MMA_XYZofLAMBDA_FromGPU_Kernel(
    PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ p0,
    const PetscScalar* __restrict__ q0,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ alpha,
    const PetscScalar* __restrict__ beta,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    const PetscScalar* __restrict__ d_lam,
    PetscInt n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar lam = d_lam[0];
    PetscScalar pjlam = p0[i] + pij[i] * lam;
    PetscScalar qjlam = q0[i] + qij[i] * lam;

    PetscScalar sqp = sqrt(pjlam);
    PetscScalar sqq = sqrt(qjlam);

    PetscScalar xi = (sqp * L[i] + sqq * U[i]) / (sqp + sqq);

    // 限制到[alpha, beta]
    if (xi < alpha[i]) xi = alpha[i];
    if (xi > beta[i]) xi = beta[i];

    x[i] = xi;
}

/**
 * GPU内核：MMA DualGrad - 计算对偶梯度的部分和
 */
__global__ void MMA_DualGrad_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    PetscScalar val = 0.0;
    if (i < n) {
        PetscScalar UmX = U[i] - x[i];
        PetscScalar XmL = x[i] - L[i];
        val = pij[i] / UmX + qij[i] / XmL;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：MMA DualHess - 计算df2和PQ
 */
__global__ void MMA_DualHess_df2PQ_Kernel(
    PetscScalar* __restrict__ df2,
    PetscScalar* __restrict__ PQ,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ p0,
    const PetscScalar* __restrict__ q0,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ alpha,
    const PetscScalar* __restrict__ beta,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    PetscInt n,
    PetscScalar lam)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    PetscScalar xi = x[i];
    PetscScalar Li = L[i];
    PetscScalar Ui = U[i];
    PetscScalar UmX = Ui - xi;
    PetscScalar XmL = xi - Li;

    PetscScalar pjlam = p0[i] + pij[i] * lam;
    PetscScalar qjlam = q0[i] + qij[i] * lam;

    // 计算PQ
    PQ[i] = pij[i] / (UmX * UmX) - qij[i] / (XmL * XmL);

    // 计算df2
    PetscScalar df2_val = -1.0 / (2.0 * pjlam / (UmX * UmX * UmX) + 2.0 * qjlam / (XmL * XmL * XmL));

    // 检查是否在边界
    PetscScalar sqp = sqrt(pjlam);
    PetscScalar sqq = sqrt(qjlam);
    PetscScalar xp = (sqp * Li + sqq * Ui) / (sqp + sqq);
    if (xp < alpha[i] || xp > beta[i]) {
        df2_val = 0.0;
    }

    df2[i] = df2_val;
}

/**
 * GPU内核：MMA DualHess - 计算Hessian元素（单约束情况）
 */
__global__ void MMA_DualHess_Hess_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ df2,
    const PetscScalar* __restrict__ PQ,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    PetscScalar val = 0.0;
    if (i < n) {
        val = PQ[i] * df2[i] * PQ[i];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * GPU内核：MMA DualResidual - 计算残差的部分和
 */
__global__ void MMA_DualResidual_Kernel(
    PetscScalar* __restrict__ partial_sums,
    const PetscScalar* __restrict__ pij,
    const PetscScalar* __restrict__ qij,
    const PetscScalar* __restrict__ x,
    const PetscScalar* __restrict__ L,
    const PetscScalar* __restrict__ U,
    PetscInt n)
{
    extern __shared__ PetscScalar sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    PetscScalar val = 0.0;
    if (i < n) {
        PetscScalar UmX = U[i] - x[i];
        PetscScalar XmL = x[i] - L[i];
        val = pij[i] / UmX + qij[i] / XmL;
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// ============================================================================
// MMA优化器C接口函数
// ============================================================================

/**
 * GPU版本的MMA初始化渐近线
 */
PetscErrorCode MMAGPU_InitAsymptotes(
    PetscScalar* d_L,
    PetscScalar* d_U,
    const PetscScalar* d_x,
    const PetscScalar* d_xmin,
    const PetscScalar* d_xmax,
    PetscInt n,
    PetscScalar asyminit)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    MMA_InitAsymptotes_Kernel<<<numBlocks, blockSize>>>(
        d_L, d_U, d_x, d_xmin, d_xmax, n, asyminit);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MMA更新渐近线
 */
PetscErrorCode MMAGPU_UpdateAsymptotes(
    PetscScalar* d_L,
    PetscScalar* d_U,
    const PetscScalar* d_x,
    const PetscScalar* d_xo1,
    const PetscScalar* d_xo2,
    const PetscScalar* d_xmin,
    const PetscScalar* d_xmax,
    PetscInt n,
    PetscScalar asymdec,
    PetscScalar asyminc)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    MMA_UpdateAsymptotes_Kernel<<<numBlocks, blockSize>>>(
        d_L, d_U, d_x, d_xo1, d_xo2, d_xmin, d_xmax, n, asymdec, asyminc);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MMA计算alpha, beta, p0, q0
 */
PetscErrorCode MMAGPU_ComputeAlphaBetaPQ(
    PetscScalar* d_alpha,
    PetscScalar* d_beta,
    PetscScalar* d_p0,
    PetscScalar* d_q0,
    const PetscScalar* d_x,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    const PetscScalar* d_xmin,
    const PetscScalar* d_xmax,
    const PetscScalar* d_dfdx,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    MMA_ComputeAlphaBetaPQ_Kernel<<<numBlocks, blockSize>>>(
        d_alpha, d_beta, d_p0, d_q0, d_x, d_L, d_U, d_xmin, d_xmax, d_dfdx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MMA计算pij和qij
 */
PetscErrorCode MMAGPU_ComputePijQij(
    PetscScalar* d_pij,
    PetscScalar* d_qij,
    const PetscScalar* d_x,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    const PetscScalar* d_dgdx,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    MMA_ComputePijQij_Kernel<<<numBlocks, blockSize>>>(
        d_pij, d_qij, d_x, d_L, d_U, d_dgdx, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MMA计算b
 */
PetscErrorCode MMAGPU_ComputeB(
    PetscScalar* result,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_x,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    MMA_ComputeB_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_pij, d_qij, d_x, d_L, d_U, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    PetscScalar* h_partial_sums = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial_sums[i];
    }
    *result = sum;

    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU版本的MMA XYZofLAMBDA
 */
PetscErrorCode MMAGPU_XYZofLAMBDA(
    PetscScalar* d_x,
    const PetscScalar* d_p0,
    const PetscScalar* d_q0,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_alpha,
    const PetscScalar* d_beta,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n,
    PetscScalar lam)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    MMA_XYZofLAMBDA_Kernel<<<numBlocks, blockSize>>>(
        d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, n, lam);
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}

/**
 * GPU版本的MMA DualGrad
 */
PetscErrorCode MMAGPU_DualGrad(
    PetscScalar* result,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_x,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    MMA_DualGrad_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_pij, d_qij, d_x, d_L, d_U, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    PetscScalar* h_partial_sums = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial_sums[i];
    }
    *result = sum;

    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU版本的MMA DualHess（单约束情况）
 */
PetscErrorCode MMAGPU_DualHess(
    PetscScalar* result,
    PetscScalar* d_df2,
    PetscScalar* d_PQ,
    const PetscScalar* d_x,
    const PetscScalar* d_p0,
    const PetscScalar* d_q0,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_alpha,
    const PetscScalar* d_beta,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n,
    PetscScalar lam)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 计算df2和PQ
    MMA_DualHess_df2PQ_Kernel<<<numBlocks, blockSize>>>(
        d_df2, d_PQ, d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, n, lam);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算Hessian元素
    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    MMA_DualHess_Hess_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_df2, d_PQ, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    PetscScalar* h_partial_sums = new PetscScalar[numBlocks];
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums,
                          numBlocks * sizeof(PetscScalar), cudaMemcpyDeviceToHost));

    PetscScalar sum = 0.0;
    for (int i = 0; i < numBlocks; i++) {
        sum += h_partial_sums[i];
    }
    *result = sum;

    delete[] h_partial_sums;
    cudaFree(d_partial_sums);

    return 0;
}

// ============================================================================
// MMA SolveDIP GPU内核（完全在GPU上执行）
// ============================================================================

/**
 * GPU内核：DualGrad直接写入GPU数组
 */
PetscErrorCode MMAGPU_DualGrad_ToGPU(
    PetscScalar* d_grad,        // 输出：GPU上的grad数组
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_x,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    MMA_DualGrad_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_pij, d_qij, d_x, d_L, d_U, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 继续归约到单个值
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        PetscScalar* d_partial2;
        CUDA_CHECK(cudaMalloc(&d_partial2, newNumBlocks * sizeof(PetscScalar)));
        ReduceSum_Kernel<<<newNumBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
            d_partial2, d_partial_sums, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_partial_sums);
        d_partial_sums = d_partial2;
        numBlocks = newNumBlocks;
    }

    // 复制结果到d_grad[0]
    CUDA_CHECK(cudaMemcpy(d_grad, d_partial_sums, sizeof(PetscScalar), cudaMemcpyDeviceToDevice));
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU内核：DualHess直接写入GPU数组
 */
PetscErrorCode MMAGPU_DualHess_ToGPU(
    PetscScalar* d_Hess,        // 输出：GPU上的Hess数组
    PetscScalar* d_df2,
    PetscScalar* d_PQ,
    const PetscScalar* d_x,
    const PetscScalar* d_p0,
    const PetscScalar* d_q0,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_alpha,
    const PetscScalar* d_beta,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscInt n,
    PetscScalar lam)
{
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // 计算df2和PQ
    MMA_DualHess_df2PQ_Kernel<<<numBlocks, blockSize>>>(
        d_df2, d_PQ, d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, n, lam);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算Hessian元素
    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    MMA_DualHess_Hess_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
        d_partial_sums, d_df2, d_PQ, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 继续归约
    while (numBlocks > 1) {
        int newNumBlocks = (numBlocks + blockSize * 2 - 1) / (blockSize * 2);
        PetscScalar* d_partial2;
        CUDA_CHECK(cudaMalloc(&d_partial2, newNumBlocks * sizeof(PetscScalar)));
        ReduceSum_Kernel<<<newNumBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
            d_partial2, d_partial_sums, numBlocks);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaFree(d_partial_sums);
        d_partial_sums = d_partial2;
        numBlocks = newNumBlocks;
    }

    // 复制结果到d_Hess[0]
    CUDA_CHECK(cudaMemcpy(d_Hess, d_partial_sums, sizeof(PetscScalar), cudaMemcpyDeviceToDevice));
    cudaFree(d_partial_sums);

    return 0;
}

/**
 * GPU内核：SolveDIP的标量运算（m=1情况）
 * 执行：grad更新、Factorize/Solve、s计算、LineSearch
 */
__global__ void MMA_SolveDIP_ScalarOps_Kernel(
    PetscScalar* __restrict__ d_lam,
    PetscScalar* __restrict__ d_mu,
    PetscScalar* __restrict__ d_grad,
    PetscScalar* __restrict__ d_Hess,
    PetscScalar* __restrict__ d_s,
    PetscScalar* __restrict__ d_y,
    PetscScalar* __restrict__ d_z,
    const PetscScalar* __restrict__ d_a,
    const PetscScalar* __restrict__ d_b,
    const PetscScalar* __restrict__ d_c,
    PetscScalar epsi)
{
    // 只需要一个线程
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    PetscScalar lam = d_lam[0];
    PetscScalar mu = d_mu[0];
    PetscScalar grad = d_grad[0];
    PetscScalar Hess = d_Hess[0];
    PetscScalar a = d_a[0];
    PetscScalar b = d_b[0];
    PetscScalar c = d_c[0];

    // 更新y和z
    if (lam < 0.0) lam = 0.0;
    PetscScalar y = max(0.0, lam - c);
    PetscScalar lamai = lam * a;
    PetscScalar z = max(0.0, 10.0 * (lamai - 1.0));

    // 完成grad计算
    grad = grad - b - a * z - y;

    // 更新grad（Newton方向）
    grad = -grad - epsi / lam;

    // 应用Hess修正
    if (lam > c) {
        Hess += -1.0;
    }
    Hess += -mu / lam;
    if (lamai > 0.0) {
        Hess += -10.0 * a * a;
    }
    PetscScalar HessCorr = 1e-4 * Hess;
    if (-HessCorr < 1.0e-7) {
        HessCorr = -1.0e-7;
    }
    Hess += HessCorr;

    // Solve（对于m=1就是除法）
    PetscScalar s0 = grad / Hess;
    PetscScalar s1 = -mu + epsi / lam - s0 * mu / lam;

    // LineSearch
    PetscScalar theta = 1.005;
    if (theta < -1.01 * s0 / lam) {
        theta = -1.01 * s0 / lam;
    }
    if (theta < -1.01 * s1 / mu) {
        theta = -1.01 * s1 / mu;
    }
    theta = 1.0 / theta;

    lam = lam + theta * s0;
    mu = mu + theta * s1;

    // 写回结果
    d_lam[0] = lam;
    d_mu[0] = mu;
    d_y[0] = y;
    d_z[0] = z;
    d_s[0] = s0;
    d_s[1] = s1;
}

/**
 * GPU内核：计算DualResidual
 */
__global__ void MMA_DualResidual_Scalar_Kernel(
    PetscScalar* __restrict__ d_err,
    PetscScalar grad_sum,
    const PetscScalar* __restrict__ d_lam,
    const PetscScalar* __restrict__ d_mu,
    const PetscScalar* __restrict__ d_a,
    const PetscScalar* __restrict__ d_b,
    const PetscScalar* __restrict__ d_y,
    const PetscScalar* __restrict__ d_z,
    PetscScalar epsi)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    PetscScalar lam = d_lam[0];
    PetscScalar mu = d_mu[0];
    PetscScalar a = d_a[0];
    PetscScalar b = d_b[0];
    PetscScalar y = d_y[0];
    PetscScalar z = d_z[0];

    PetscScalar res0 = grad_sum - b - a * z - y + mu;
    PetscScalar res1 = mu * lam - epsi;

    PetscScalar nrI = fabs(res0);
    if (fabs(res1) > nrI) nrI = fabs(res1);

    d_err[0] = nrI;
}

/**
 * GPU内核：更新y和z
 */
__global__ void MMA_UpdateYZ_Kernel(
    PetscScalar* __restrict__ d_y,
    PetscScalar* __restrict__ d_z,
    const PetscScalar* __restrict__ d_lam,
    const PetscScalar* __restrict__ d_a,
    const PetscScalar* __restrict__ d_c)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    PetscScalar lam = d_lam[0];
    if (lam < 0.0) lam = 0.0;
    d_y[0] = max(0.0, lam - d_c[0]);
    PetscScalar lamai = lam * d_a[0];
    d_z[0] = max(0.0, 10.0 * (lamai - 1.0));
}

/**
 * GPU版本的完整SolveDIP（单约束m=1情况）
 * 在GPU上执行大部分计算，MPI通信时需要CPU参与
 */
PetscErrorCode MMAGPU_SolveDIP(
    PetscScalar* d_x,           // 输入/输出：设计变量
    PetscScalar* d_lam,         // 输入/输出：拉格朗日乘子
    PetscScalar* d_mu,          // 输入/输出：对偶变量
    PetscScalar* d_y,           // 工作空间
    PetscScalar* d_z,           // 工作空间（单个标量）
    PetscScalar* d_grad,        // 工作空间
    PetscScalar* d_Hess,        // 工作空间
    PetscScalar* d_s,           // 工作空间
    PetscScalar* d_a,           // 参数
    PetscScalar* d_b,           // 参数
    PetscScalar* d_c,           // 参数
    const PetscScalar* d_p0,
    const PetscScalar* d_q0,
    const PetscScalar* d_pij,
    const PetscScalar* d_qij,
    const PetscScalar* d_alpha,
    const PetscScalar* d_beta,
    const PetscScalar* d_L,
    const PetscScalar* d_U,
    PetscScalar* d_df2,         // 工作空间
    PetscScalar* d_PQ,          // 工作空间
    PetscInt n,
    PetscInt m)
{
    // 只支持m=1
    if (m != 1) {
        PetscPrintf(PETSC_COMM_WORLD, "MMAGPU_SolveDIP only supports m=1\n");
        return PETSC_ERR_SUP;
    }

    PetscScalar tol = 1.0e-9 * sqrt((PetscScalar)(m + n));
    PetscScalar epsi = 1.0;

    // 临时变量用于归约结果
    PetscScalar* d_err;
    CUDA_CHECK(cudaMalloc(&d_err, sizeof(PetscScalar)));

    // 预分配归约用的临时数组
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    PetscScalar* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, numBlocks * sizeof(PetscScalar)));

    while (epsi > tol) {
        PetscScalar err = 1.0;
        int loop = 0;

        while (err > 0.9 * epsi && loop < 100) {
            loop++;

            // 1. 更新y和z（从GPU数组读取lam）
            MMA_UpdateYZ_Kernel<<<1, 1>>>(d_y, d_z, d_lam, d_a, d_c);

            // 2. XYZofLAMBDA - 使用从GPU数组读取lam的版本
            MMA_XYZofLAMBDA_FromGPU_Kernel<<<numBlocks, blockSize>>>(
                d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, d_lam, n);

            // 3. DualGrad - 本地归约
            MMA_DualGrad_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
                d_partial_sums, d_pij, d_qij, d_x, d_L, d_U, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 本地归约
            int nb = numBlocks;
            PetscScalar* d_current = d_partial_sums;
            PetscScalar* d_temp = NULL;
            while (nb > 1) {
                int newNb = (nb + blockSize * 2 - 1) / (blockSize * 2);
                CUDA_CHECK(cudaMalloc(&d_temp, newNb * sizeof(PetscScalar)));
                ReduceSum_Kernel<<<newNb, blockSize, blockSize * sizeof(PetscScalar)>>>(
                    d_temp, d_current, nb);
                CUDA_CHECK(cudaDeviceSynchronize());
                if (d_current != d_partial_sums) cudaFree(d_current);
                d_current = d_temp;
                nb = newNb;
            }
            PetscScalar grad_local;
            CUDA_CHECK(cudaMemcpy(&grad_local, d_current, sizeof(PetscScalar), cudaMemcpyDeviceToHost));
            if (d_current != d_partial_sums) cudaFree(d_current);

            // MPI归约
            PetscScalar grad_global;
            MPI_Allreduce(&grad_local, &grad_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(d_grad, &grad_global, sizeof(PetscScalar), cudaMemcpyHostToDevice));

            // 4. DualHess - 需要lam值
            PetscScalar lam_host;
            CUDA_CHECK(cudaMemcpy(&lam_host, d_lam, sizeof(PetscScalar), cudaMemcpyDeviceToHost));

            // 计算df2和PQ
            MMA_DualHess_df2PQ_Kernel<<<numBlocks, blockSize>>>(
                d_df2, d_PQ, d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, n, lam_host);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 计算Hessian元素（本地归约）
            MMA_DualHess_Hess_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
                d_partial_sums, d_df2, d_PQ, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            nb = numBlocks;
            d_current = d_partial_sums;
            d_temp = NULL;
            while (nb > 1) {
                int newNb = (nb + blockSize * 2 - 1) / (blockSize * 2);
                CUDA_CHECK(cudaMalloc(&d_temp, newNb * sizeof(PetscScalar)));
                ReduceSum_Kernel<<<newNb, blockSize, blockSize * sizeof(PetscScalar)>>>(
                    d_temp, d_current, nb);
                CUDA_CHECK(cudaDeviceSynchronize());
                if (d_current != d_partial_sums) cudaFree(d_current);
                d_current = d_temp;
                nb = newNb;
            }
            PetscScalar hess_local;
            CUDA_CHECK(cudaMemcpy(&hess_local, d_current, sizeof(PetscScalar), cudaMemcpyDeviceToHost));
            if (d_current != d_partial_sums) cudaFree(d_current);

            // MPI归约
            PetscScalar hess_global;
            MPI_Allreduce(&hess_local, &hess_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(d_Hess, &hess_global, sizeof(PetscScalar), cudaMemcpyHostToDevice));

            // 5. 标量运算（grad更新、Solve、LineSearch）- 完全在GPU上
            MMA_SolveDIP_ScalarOps_Kernel<<<1, 1>>>(
                d_lam, d_mu, d_grad, d_Hess, d_s, d_y, d_z, d_a, d_b, d_c, epsi);

            // 6. 再次更新y和z，然后XYZofLAMBDA（使用GPU版本）
            MMA_UpdateYZ_Kernel<<<1, 1>>>(d_y, d_z, d_lam, d_a, d_c);
            MMA_XYZofLAMBDA_FromGPU_Kernel<<<numBlocks, blockSize>>>(
                d_x, d_p0, d_q0, d_pij, d_qij, d_alpha, d_beta, d_L, d_U, d_lam, n);

            // 7. DualResidual - 计算grad_sum
            MMA_DualGrad_Kernel<<<numBlocks, blockSize, blockSize * sizeof(PetscScalar)>>>(
                d_partial_sums, d_pij, d_qij, d_x, d_L, d_U, n);
            CUDA_CHECK(cudaDeviceSynchronize());

            // 本地归约
            nb = numBlocks;
            d_current = d_partial_sums;
            d_temp = NULL;
            while (nb > 1) {
                int newNb = (nb + blockSize * 2 - 1) / (blockSize * 2);
                CUDA_CHECK(cudaMalloc(&d_temp, newNb * sizeof(PetscScalar)));
                ReduceSum_Kernel<<<newNb, blockSize, blockSize * sizeof(PetscScalar)>>>(
                    d_temp, d_current, nb);
                CUDA_CHECK(cudaDeviceSynchronize());
                if (d_current != d_partial_sums) cudaFree(d_current);
                d_current = d_temp;
                nb = newNb;
            }
            PetscScalar res_local;
            CUDA_CHECK(cudaMemcpy(&res_local, d_current, sizeof(PetscScalar), cudaMemcpyDeviceToHost));
            if (d_current != d_partial_sums) cudaFree(d_current);

            // MPI归约
            PetscScalar res_global;
            MPI_Allreduce(&res_local, &res_global, 1, MPIU_SCALAR, MPI_SUM, PETSC_COMM_WORLD);

            // 计算残差
            MMA_DualResidual_Scalar_Kernel<<<1, 1>>>(d_err, res_global, d_lam, d_mu, d_a, d_b, d_y, d_z, epsi);
            CUDA_CHECK(cudaMemcpy(&err, d_err, sizeof(PetscScalar), cudaMemcpyDeviceToHost));
        }

        epsi = epsi * 0.1;
    }

    cudaFree(d_partial_sums);
    cudaFree(d_err);
    return 0;
}

} // extern "C"