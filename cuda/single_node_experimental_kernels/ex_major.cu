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

/** A100: N = 15, k = 2, prior = 0.1: 89.0362 seconds */
/** RTX3060: N=13, k = 2, prior = 0.1: 10.4249 seconds */
/** RTX3060: N = 15, k = 2, prior = 0.1: 562.238 seconds */
template <int n, int k>
__global__ void BBPA(const float *_post_probs, float *mins)
{
    int partition_id = 0;
    float partition_mass[1 << k];
    memset(partition_mass, 0, (1 << k) * sizeof(float));
    int ex = blockIdx.x * blockDim.x + threadIdx.x;
    for (int s_iter = 0; s_iter < (1 << (n * k)); s_iter++)
    {
        // #pragma unroll
        for (int variant = 0; variant < k; variant++)
        {
            partition_id |= ((1 << variant) & (((ex & (s_iter >> (variant * n))) - ex) >> 31));
        }

        // partition_id |= (1 & (((ex & s_iter) - ex) >> 31));
        // partition_id |= (2 & (((ex & (s_iter >> n)) - ex) >> 31));
        partition_mass[partition_id] += _post_probs[s_iter];
        partition_id = 0;
    }

    float min = 0.0;
    for(int i = 0; i < (1 << k); i++){
        min += abs(partition_mass[i] - 1.0 / (1 << k));
    }
    mins[ex] = min;
}

int main()
{
    float *post_probs, *mins;

    std::cout << "N = " << N << ", k = " << K << ", prior = " << (float)(P) / 100.0 << std::endl;
    int numElements = (1 << (N * K));
    cudaError_t cudaStatus = cudaMalloc((void **)&post_probs, numElements * sizeof(float));

    cudaStatus = cudaMalloc((void **)&mins, (1 << K) * sizeof(float));
    // cudaStatus = cudaMemset(mins, 0, (1 << K) * sizeof(float));

    bin_enc *d_candidate;
    cudaMalloc((void **)&d_candidate, sizeof(bin_enc));

    dim3 blockDims(B);                                          // Adjust block dimensions as needed
    dim3 gridDims((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();
    set_prior_probs<N, K, P><<<gridDims, blockDims>>>(post_probs);

    end = std::chrono::system_clock::now();
    std::chrono::duration<float> elapsedSeconds = end - start;

    std::cout << "Prior kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    numElements = (1 << N);
    dim3 gridDims1((numElements + blockDims.x - 1) / blockDims.x); // Calculate grid dimensions

    start = std::chrono::system_clock::now();

    BBPA<N, K><<<gridDims1, blockDims>>>(post_probs, mins);

    cudaDeviceSynchronize(); // Wait for the kernel to finish

    end = std::chrono::system_clock::now();
    elapsedSeconds = end - start;

    std::cout << "BBPA kernel execution time: " << elapsedSeconds.count() << " seconds" << std::endl;

    // Copy the result back from the GPU
    bin_enc h_candidate;
    cudaMemcpy(&h_candidate, d_candidate, sizeof(bin_enc), cudaMemcpyDeviceToHost);

    float *h_mins = new float[(1 << N)];
    cudaMemcpy(h_mins, mins, (1 << N) * sizeof(float), cudaMemcpyDeviceToHost);
    float global_min = 2.0;
    float global_candidate = -1;
    for (int i = 0; i < (1 << N); i++)
    {
        if(h_mins[i] < global_min){
            global_min = h_mins[i];
            global_candidate = i;
        }
    }

    std::cout << "\nCandidate is: " << global_candidate << std::endl;

    // Free allocated memory on the GPU
    cudaFree(post_probs);
    cudaFree(mins);
    delete[] h_mins;

    return 0;
}
