#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

#ifndef N
#define N 13
#endif

#ifndef K
#define K 2
#endif

#ifndef P
#define P 10
#endif

#ifndef B
#define B 8
#endif

typedef int bin_enc;

template <int n, int k, int p>
__global__ void set_prior_probs(float *_post_probs)
{
    const float pi0 = p / 100.0f;
    int s_iter = blockIdx.x * blockDim.x + threadIdx.x;
    float prob = 1.0f;
    for (int i = 0; i < n * k; i++)
    {
        if ((s_iter & (1 << i)) == 0)
            prob *= pi0;
        else
            prob *= (1.0f - pi0);
    }
    _post_probs[s_iter] = prob;
}

/** A100: N = 15, k = 2, prior = 0.1: 46.9054 seconds
 *  RTX3060: N = 15, k = 2, prior = 0.1: 154.283 seconds
 */
template <int n, int k, int b>
__global__ void BBPA(const float *probs, float *mass)
{
    int laneId = threadIdx.x & 0x1f;
    float r_mass[1 << k];
    memset(r_mass, 0, (1 << k) * sizeof(float));
    int ex = blockIdx.x;
    for (int s_iter = 0; s_iter < (1 << (n * k - b)); s_iter++)
    {
        int state = threadIdx.x * (1 << (n * k - b)) + s_iter;
        int partition_id = 0;
        #pragma unroll k
        for (int variant = 0; variant < k; variant++)
        {
            partition_id |= ((1 << variant) & (((ex & (state >> (variant * n))) - ex) >> 31));
        }
        // partition_id |= (1 & (((ex & state) - ex) >> 31));
        // partition_id |= (2 & (((ex & (state >> n)) - ex) >> 31));
        r_mass[partition_id] += probs[state];
    }

    for (int i = 0; i < (1 << k); i++)
    {
        for (int j = 16; j >= 1; j /= 2)
        {
            r_mass[i] += __shfl_xor_sync(0xffffffff, r_mass[i], j, 32);
        }
    }
    if(!laneId){
        atomicAdd(&mass[blockIdx.x], r_mass[0] + r_mass[1] + r_mass[2] + r_mass[3]);
    }
}

int main()
{
    float *d_probs, *d_mass;

    std::cout << "N = " << N << ", k = " << K << ", prior = " << P / 100.0f << std::endl;
    int numElements = (1 << (N * K));
    cudaError_t cudaStatus = cudaMalloc((void **)&d_probs, numElements * sizeof(float));

    cudaStatus = cudaMalloc((void **)&d_mass, (1 << (N + K)) * sizeof(float));
    // cudaStatus = cudaMemset(mins, 0, (1 << curr_subjs) * sizeof(float));

    bin_enc *d_candidate;
    cudaMalloc((void **)&d_candidate, sizeof(bin_enc));

    dim3 blockDims(1 << B);                                       // Adjust block dimensions as needed
    dim3 gridDims((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    set_prior_probs<N, K, P><<<gridDims, blockDims>>>(d_probs);

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsedSeconds = end - start;

    std::cout << "Prior kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    dim3 gridDims1(1 << N); // Calculate grid dimensions

    start = std::chrono::system_clock::now();

    BBPA<N, K, B><<<gridDims1, blockDims>>>(d_probs, d_mass);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    end = std::chrono::system_clock::now();
    elapsedSeconds = end - start;

    std::cout << "BBPA kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    // Copy the result back from the GPU
    float *h_partition_mass = new float[1 << (N + K)];
    cudaMemcpy(h_partition_mass, d_mass, (1 << (N + K)) * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; i ++)
    {
        std::cout << h_partition_mass[i] << "  ";
    }
    std::cout << std::endl;

    // Free allocated memory on the GPU
    cudaFree(d_probs);
    cudaFree(d_mass);

    return 0;
}
