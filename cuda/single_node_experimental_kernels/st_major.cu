#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#ifndef NUM
#define NUM 13
#endif

#ifndef K
#define K 2
#endif

#ifndef SMEM
#define SMEM 9
#endif

#ifndef BLOCK
#define BLOCK 256
#endif

typedef int bin_enc;

template <int N, int k, int prior_numer>
__global__ void set_prior_probs(float *_post_probs)
{
    const float pi0 = (float)(prior_numer) / 100.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float prob = 1.0f;
    for (int i = 0; i < N * k; i++)
    {
        if ((tid & (1 << i)) == 0)
            prob *= pi0;
        else
            prob *= (1.0f - pi0);
    }
    _post_probs[tid] = prob;
}

/** RTX3060: N = 11, k = 2, prior = 0.3, block 256: 0.198432 seconds
 *  A100: N = 15, k = 2, prior = 0.1, block 1024: 30.4172 seconds
*/
template <int N, int k>
__global__ void BBPA_target(const float *__restrict__ _post_probs, float *__restrict__ partition_mass)
{
    int partition_id = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = _post_probs[tid];
    for (bin_enc ex = 0; ex < (1 << N); ex++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < k; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
        // }

        partition_id |= (1 & (((ex & tid) - ex) >> 31));
        partition_id |= (2 & (((ex & (tid >> N)) - ex) >> 31));
        // atomicAdd(&partition_mass[ex * (1 << k) + partition_id], val);
        partition_mass[ex * (1 << k) + partition_id] = val;
        partition_id = 0;
    }
}

/** RTX3060: N = 11, k = 2, prior = 0.3 block 256: 0.844862 seconds
 *  A100: N = 15, k = 2, prior = 0.1 blick 256: 736.655 seconds
*/
template <int N, int k>
__global__ void BBPA_write_aligned(const float *__restrict__ _post_probs, float *__restrict__ partition_mass)
{
    int partition_id = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = _post_probs[tid];
    for (bin_enc ex = 0; ex < (1 << N); ex++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < k; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
        // }

        partition_id |= (1 & (((ex & tid) - ex) >> 31));
        partition_id |= (2 & (((ex & (tid >> N)) - ex) >> 31));
        atomicAdd(&partition_mass[ex * (1 << k) + partition_id], val);
        // partition_mass[ex * (1 << k) + partition_id] += val;
        partition_id = 0;
    }
}

/** N = 11, k = 2, prior = 0.3 block 256: 0.580698 seconds
 *  N = 15, k = 2, prior = 0.1 block 256: 335.417 seconds
*/
template <int N, int k>
__global__ void BBPA(const float *__restrict__ _post_probs, float *__restrict__ partition_mass)
{
    int partition_id = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = _post_probs[tid];
    for (bin_enc experiment = threadIdx.x; experiment < (1 << N) + threadIdx.x; experiment++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < k; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
        // }

        int ex = experiment % (1 << N);
        partition_id |= (1 & (((ex & tid) - ex) >> 31));
        partition_id |= (2 & (((ex & (tid >> N)) - ex) >> 31));
        atomicAdd(&partition_mass[ex * (1 << k) + partition_id], val);
        // partition_mass[experiment * (1 << k) + partition_id] += val;
        partition_id = 0;
    }
}

/** N = 11, k = 2, prior = 0.3, block 256: 0.0725678 seconds*/
template <int N, int k>
__global__ void BBPA_smem_interleave(const float *__restrict__ _post_probs, float *__restrict__ partition_mass)
{
    __shared__ float block_partition_mass[(1 << N) * (1 << k)];
    if (threadIdx.x == 0)
        memset(block_partition_mass, 0, (1 << N) * (1 << k) * sizeof(float));

    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (bin_enc experiment = threadIdx.x; experiment < (1 << N) + threadIdx.x; experiment++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < k; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
        // }
        int ex = experiment % (1 << N);

        int partition_id = ((1 & (((ex & tid) - ex) >> 31))) | ((2 & (((ex & (tid >> N)) - ex) >> 31)));
        block_partition_mass[ex * (1 << k) + partition_id] += _post_probs[tid];
        __syncthreads();
    }
    // __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < (1 << N) * (1 << k); i++)
        {
            atomicAdd(&partition_mass[i], block_partition_mass[i]);
        }
    }
}

/** RTX3060: N = 11, k = 2, prior = 0.3, block 256: 0.0725678 seconds
 *  A100: N = 15, k = 2, prior = 0.3, block 256: 82.3447 seconds
 *  A100: N = 15, k = 2, prior = 0.3, block 256: 69.4607 seconds
*/
template <int N, int k, int smem>
__global__ void BBPA_smem_interleave(const float *__restrict__ _post_probs, float *__restrict__ partition_mass)
{
    __shared__ float block_partition_mass[(1 << (smem + k))];
    if (threadIdx.x == 0)
        memset(block_partition_mass, 0, (1 << (smem + k)) * sizeof(float));

    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int iter = 0; iter < (1 << (N - smem)); iter++)
    {
        for (bin_enc experiment = threadIdx.x; experiment < (1 << smem) + threadIdx.x; experiment++)
        {
            // #pragma unroll
            // for (int variant = 0; variant < k; variant++)
            // {
            //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
            // }
            int ex = (experiment % (1 << smem)) + iter * (1 << smem);

            int partition_id = ((1 & (((ex & tid) - ex) >> 31))) | ((2 & (((ex & (tid >> N)) - ex) >> 31)));
            block_partition_mass[(experiment % (1 << smem)) * (1 << k) + partition_id] += _post_probs[tid];
            __syncthreads();
        }
        // __syncthreads();
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < (1 << (smem + k)); i++)
            {
                atomicAdd(&partition_mass[iter * (1 << (smem + k)) + i], block_partition_mass[i]);
            }
            memset(block_partition_mass, 0, (1 << (smem + k)) * sizeof(float));
        }
        __syncthreads();
    }
}

/** N = 11, k = 2, prior = 0.3, block 256, BBPA kernel execution time: 0.829783 seconds*/
template <int N, int k>
__global__ void BBPA_smem(const float *__restrict__ _post_probs, float *partition_mass)
{
    __shared__ float block_partition_mass[(1 << N) * (1 << k)];
    if (threadIdx.x == 0)
        memset(block_partition_mass, 0, (1 << N) * (1 << k) * sizeof(float));

    __syncthreads();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (bin_enc ex = 0; ex < (1 << N); ex++)
    {
        // #pragma unroll
        // for (int variant = 0; variant < k; variant++)
        // {
        //     partition_id |= ((1 << variant) & (((experiment & (tid >> (variant * N))) - experiment) >> 31));
        // }

        int partition_id = ((1 & (((ex & tid) - ex) >> 31))) | ((2 & (((ex & (tid >> N)) - ex) >> 31)));
        // __syncthreads();
        atomicAdd(&block_partition_mass[ex * (1 << k) + partition_id], _post_probs[tid]);
        // __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < (1 << N) * (1 << k); i++)
        {
            atomicAdd(&partition_mass[i], block_partition_mass[i]);
        }
    }
}

int main()
{
    float *post_probs, *partition_mass;
    constexpr int prior_numer = 10;
    std::cout << "N = " << NUM << ", k = " << K << ", prior = " << (float)(prior_numer) / 100.0 << std::endl;
    int numElements = (1 << (NUM * K));
    cudaError_t cudaStatus = cudaMalloc((void **)&post_probs, numElements * sizeof(float));

    cudaStatus = cudaMalloc((void **)&partition_mass, (1 << NUM) * (1 << K) * sizeof(float));
    cudaStatus = cudaMemset(partition_mass, 0, (1 << NUM) * (1 << K) * sizeof(float));

    bin_enc *d_candidate;
    cudaMalloc((void **)&d_candidate, sizeof(bin_enc));

    dim3 blockDims(BLOCK);                                          // Adjust block dimensions as needed
    dim3 gridDims((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions
    static_assert(BLOCK <= (1 << SMEM), "Allocated shared memory too small!\n");

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    set_prior_probs<NUM, K, prior_numer><<<gridDims, blockDims>>>(post_probs);

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsedSeconds = end - start;

    std::cout << "Prior kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    start = std::chrono::system_clock::now();

    BBPA_smem_interleave<NUM, K, SMEM><<<gridDims, blockDims>>>(post_probs, partition_mass);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    end = std::chrono::system_clock::now();
    elapsedSeconds = end - start;

    std::cout << "BBPA_smem_interleave time: " << elapsedSeconds.count() << " seconds" << std::endl;


    // start = std::chrono::system_clock::now();

    // BBPA_target<NUM, K><<<gridDims, blockDims>>>(post_probs, partition_mass);

    // cudaDeviceSynchronize(); // Wait for the kernel to finish

    // end = std::chrono::system_clock::now();
    // elapsedSeconds = end - start;

    // std::cout << "BBPA_target time: " << elapsedSeconds.count() << " seconds" << std::endl;


    // Copy the result back from the GPU
    bin_enc h_candidate;
    cudaMemcpy(&h_candidate, d_candidate, sizeof(bin_enc), cudaMemcpyDeviceToHost);

    float *h_partition_mass = new float[(1 << NUM) * (1 << K)];
    cudaMemcpy(h_partition_mass, partition_mass, (1 << NUM) * (1 << K) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 40; i += 4)
    {
        std::cout << h_partition_mass[i] + h_partition_mass[i + 1] + h_partition_mass[i + 2] + h_partition_mass[i + 3] << "  ";
    }

    std::cout << "\nCandidate is: " << h_candidate << std::endl;

    // Free allocated memory on the GPU
    cudaFree(post_probs);
    cudaFree(partition_mass);
    cudaFree(d_candidate);

    return 0;
}
